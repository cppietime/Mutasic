'''Message-based music generation'''

from dataclasses import dataclass, field
import enum, math, random, typing

import mido
import numpy as np

from mutasic import musicrep

major_scale = [0, 2, 4, 5, 7, 9, 11]
minor_scale = [0, 2, 3, 5, 7, 8, 10]
harmonic_minor_scale = [0, 2, 3, 5, 7, 8, 11]
chromatic_scale = tuple(range(12))
pent_major = [0, 2, 4, 7, 9]
pent_minor = [0, 3, 5, 7, 10]
blues_hex_major = [0, 2, 3, 4, 7, 9]
blues_hex_minor = [0, 3, 5, 6, 7, 10]
# Uncomment me to test microtonality
blues_hex_minor = [0, 0.25, 0.75, 1.33, 1.75]
dissonance = [0,
    math.log(16 * 15),
    math.log(9 * 8),
    math.log(6 * 5),
    math.log(5 * 4),
    math.log(4 * 3),
    math.log(45 * 32),
    math.log(3 * 2),
    math.log(8 * 5),
    math.log(5 * 3),
    math.log(9 * 5),
    math.log(15 * 8)]

class ScaleModes(enum.IntEnum):
    IONIAN = 0
    MAJOR = 0
    DORIAN = 1
    PHRYGIAN = 2
    LYDIAN = 3
    MIXOLYDIAN = 4
    AEIOLEAN = 5
    MINOR = 5
    LOCRIAN = 6 # The forbidden jelly

class Note(enum.IntEnum):
    C = 0
    C_SHARP = 1
    D_FLAT = 1
    D = 2
    D_SHARP = 3
    E_FLAT = 3
    E = 4
    F = 5
    F_SHARP = 6
    G_FLAT = 6
    G = 7
    G_SHARP = 8
    A_FLAT = 8
    A = 9
    A_SHARP = 10
    B_FLAT = 10
    B = 11

def mode_of_scale(scale, mode):
    if mode < 0 or mode >= len(scale):
        raise ValueError(f'{mode} not within range of provided scale')
    if mode == 0:
        return scale
    intervals = [scale[i] - scale[i - 1] for i in range(1, len(scale))]
    intervals.append(12 - sum(intervals))
    intervals = intervals[mode:] + intervals[:mode]
    scale = [0]
    for i in intervals[:-1]:
        scale.append(scale[-1] + i)
    return scale

def extend_octaves(scale, octaves):
    return [note for octave in map(lambda x: map(lambda z: z + x * 12, scale), range(octaves)) for note in octave]

def create_scale(base_note = Note.C, mode = ScaleModes.IONIAN, scale = major_scale, octaves = 1):
    scale = mode_of_scale(scale, mode)
    scale = map(lambda x: x + base_note, scale)
    scale = extend_octaves(scale, octaves)
    return scale

def calc_dissonance(note0, note1):
    interval = abs(note1 - note0) % 12
    interval = min(interval, 12 - interval)
    if float(interval).is_integer():
        return dissonance[int(interval)]
    return 0

def geom(p):
    i = 0
    while True:
        if random.random() <= p:
            break
        i += 1
    return i

def order_chords(relative_to, choices):
    chords = []
    for choice in choices:
        energy = 0
        for note0 in choice:
            for note1 in relative_to:
                energy += calc_dissonance(note0, note1)
            for note1 in choice:
                energy += calc_dissonance(note0, note1)
        chords.append((energy / len(choice), choice))
    chords.sort()
    return chords

def all_chords(length, out_of = range(12)):
    if isinstance(length, tuple):
        chords = []
        for i in range(length[0], length[1] + 1):
            chords += all_chords(i, out_of = out_of)
        print(f'{chords=}{length=}')
        return chords
    else:
        maxLen = length
        nexLen = maxLen - 1
    if maxLen == 1:
        return [ tuple([i]) for i in out_of ]
    chords = []
    for base in range(len(out_of)):
        trunc = all_chords(nexLen, out_of[base + 1:])
        chords += tuple([ tuple([out_of[base]]) + chord for chord in trunc ])
    return chords

@dataclass
class Voice():
    '''One of one or more voices for melodies'''
    channel: int
    count: int = 1
    base_note: int = 12
    notes: typing.Sequence[typing.Union[int, typing.Sequence[int]]] = None

@dataclass
class SongBuildingContext():
    '''A context with parameters for song building'''
    notes: typing.Sequence[int] = field(default_factory = lambda: major_scale)
    chords: typing.Sequence[typing.Sequence[int]] = field(default_factory = lambda: None)
    drum_beat_offset: int = 0
    jazz: typing.Sequence[float] = 1/3, 1/10 # On beat, off beat. Lower value = higher jazziness
    base_note: int = 64
    k_notes_per_measure: int = 4 # 2^k notes per measure
    k_measure_div: int = 2 # 2^k submeasures per supermeasure
    no_drum: int = -1 # Supress drums
    rest: typing.Sequence[float] = (.25, .75) # Chance of resting on and off beat
    drum_notes: typing.Sequence[int] = (36, )
    chord_instability: float = 1 # Lower = more chord changes
    base_mutation_rate: float = 0 # Higher = more note changes
    voices: typing.Sequence[Voice] = field(default_factory = lambda: [Voice(1, 1, 0), Voice(2, 1, 0)])
    
    def __post_init__(self):
        self.microtonal = any(map(lambda x: not float(x).is_integer(), tuple(self.notes) + sum(map(tuple, self.chords), ())))
        self.percussion = []
        if self.chords is None:
            self.chords = all_chords(3, out_of = self.notes)
        self.notes = [[i] for i in self.notes]
        for voice in self.voices:
            if voice.notes is None:
                voice.notes = self.notes
            else:
                voice.notes = [[i] for i in voice.notes]
    
    def random_drums(self):
        index = random.randint(0, len(self.percussion))
        if index == len(self.percussion):
            drums = []
            for i in range(1 << self.k_notes_per_measure):
                i = (i - self.drum_beat_offset) % (1 << self.k_notes_per_measure)
                if i == 0:
                    i = 1 << (self.k_notes_per_measure - 1)
                i = i & (-i)
                bits = i.bit_length() - 1 # 8 - 3, 4 - 2, 2 - 1, 1 - 0
                bits = min(len(self.drum_notes), bits)
                drum = random.randint(0, (1 << (self.k_notes_per_measure - 1)) + self.no_drum) <= i
                drums.append(self.drum_notes[len(self.drum_notes) - 1 - bits] if drum else 0)
            self.percussion.append(drums)
        return index
    
    def mutate_drums(self, index, starting_at = 8):
        drums = list(self.percussion[index])
        for i in range(starting_at, len(drums)):
            j = i
            i = (i - self.drum_beat_offset) % (1 << self.k_notes_per_measure)
            if i == 0:
                i = 1 << (self.k_notes_per_measure - 1)
            i = i & (-i)
            bits = i.bit_length() - 1 # 8 - 3, 4 - 2, 2 - 1, 1 - 0
            drum = random.randint(0, (1 << (self.k_notes_per_measure - 1)) + self.no_drum) <= i
            bits = min(len(self.drum_notes), bits)
            drums[j] = self.drum_notes[len(self.drum_notes) - 1 - bits] if drum else 0
        self.percussion.append(drums)
        return len(self.percussion) - 1

class Measure():
    '''One measure/unit in a musical composition.
    If depth==0, this holds notes, otherwise, it holds smaller measures'''
    def __init__(self, depth, base, context, children = None, percussion = None):
        self.depth = depth
        self.context = context
        self.shift = [None] * (1 << self.context.k_notes_per_measure)
        self.base = base
        self.mutant = False
        if children is not None:
            self.base = base
            self.children = children
            self.percussion = percussion
        else:
            self.percussion = context.random_drums()
            self.children = [[set() for _ in context.voices]  for _ in range(1 << context.k_notes_per_measure)]
            if self.depth == 0:
                self.decide_notes()
            else:
                first = Measure(depth - 1, self.base, context)
                self.children = [first]
                for i in range(1, 1 << context.k_measure_div):
                    self.children.append(first.mutate(context.base_mutation_rate + (1 - context.base_mutation_rate) * i / ((1 << context.k_measure_div) - 1)))
    
    def decide_notes(self, starting_at = 0):
        if self.depth != 0:
            raise Exception("Calling note decision on parent measure")
        curChord = self.base
        for i in range(int(random.random() / (self.context.chord_instability))):
            index = 1 << (1 + random.randint(0, self.context.k_notes_per_measure - 2))
            self.shift[index] = []
        for i in range(starting_at, len(self.children)):
            p = self.context.jazz[0] if self.context.percussion[self.percussion][i] != 0 else self.context.jazz[1]
            if self.shift[i] is not None:
                curChord = self.shift[i]
                ordered_chords = order_chords(self.base, self.context.chords)
                chord = geom(p) % len(ordered_chords)
                self.shift[i] = ordered_chords[chord][1]
            playing = set(curChord)
            for j, voice in enumerate(self.context.voices):
                voiceNotes = set()
                for k in range(voice.count):
                    nOrdered = order_chords(playing, voice.notes)
                    rest = self.context.rest[0] if self.context.percussion[self.percussion][i] != 0 else self.context.rest[1]
                    if (i == 0 and k == 0) or random.random() >= rest:
                        note = geom(p) % len(nOrdered)
                        voiceNotes = voiceNotes.union(nOrdered[note][1])
                    elif i > 0 and k == 0:
                        voiceNotes = voiceNotes.union(self.children[i - 1][j])
                    playing = playing.union(voiceNotes)
                self.children[i][j] = voiceNotes
    
    def _mark_non_mutated(self):
        self.mutant = False
        for child in self.children:
            child._mark_non_mutated()
    
    def mutate(self, p):
        self.mutant = True
        self = self.copy()
        if self.depth == 0:
            if random.random() <= p:
                # starting_point = (1 << self.context.k_notes_per_measure) - (1 << random.randint(1, self.context.k_notes_per_measure))
                starting_point = random.randint(0, (1 << (self.context.k_notes_per_measure - 1)) - 1) << 1
                self.percussion = self.context.mutate_drums(self.percussion, starting_point)
                self.decide_notes(starting_point)
        else:
            ordered = order_chords(self.base, self.context.chords)
            chord = geom(self.context.jazz[0]) % len(ordered)
            self.base = ordered[chord][1]
            print(f'changed to {self.base=}')
            for i in range(1 << self.context.k_measure_div):
                self.children[i].base = self.base
                if random.random() <= p:
                    new_p = p + (1 - p) * ((self.context.k_measure_div + 1) - (i & (-i))) / (self.context.k_measure_div + 1)
                    self.children[i] = self.children[i].mutate(new_p)
                    if (i & (i + 1)) == 0:
                        for j in range(i * 2 + 1, 1 << self.context.k_measure_div, i + 1):
                            self.children[j] = self.children[i]
        return self
    
    def copy(self):
        copy_children = [list(child) for child in self.children] if self.depth == 0 else [child.copy() for child in self.children]
        copy = Measure(self.depth, self.base, self.context, copy_children, self.percussion)
        copy.shift = [None if s is None else list(s) for s in self.shift]
        copy.mutant = self.mutant
        return copy
    
    def note_on_midi(self, note, velocity, pitch_bends, free_channels, delta_time, msg_target):
        if not self.context.microtonal:
            msg_target(mido.Message('note_on', note=int(note), channel=0, velocity=velocity, time=delta_time))
            return
        pitch_bend = note - int(note)
        pitch_bend = int(pitch_bend * 4096)
        if pitch_bend in pitch_bends:
            l = pitch_bends[pitch_bend]
            channel = l[0]
            l[1] += 1
        else:
            channel = heapq.heappop(free_channels)
            msg_target(mido.Message('pitchwheel', channel=channel, pitch=pitch_bend, time=delta_time))
            delta_time = 0
            pitch_bends[pitch_bend] = [channel, 1]
        msg_target(mido.Message('note_on', note=int(note), channel=channel, velocity=velocity, time=delta_time))
    
    def note_off_midi(self, note, pitch_bends, free_channels, delta_time, msg_target):
        if not self.context.microtonal:
            msg_target(mido.Message('note_off', note=int(note), channel=0, time=delta_time))
            return
        pitch_bend = note - int(note)
        pitch_bend = int(pitch_bend * 4096)
        if pitch_bend in pitch_bends:
            l = pitch_bends[pitch_bend]
            channel = l[0]
            l[1] -= 1
            if l[1] == 0:
                pitch_bends.pop(pitch_bend)
                heapq.heappush(free_channels, channel)
        else:
            print(f'Warning: Pitch bend {pitch_bend} turned off before being turned on')
            channel = 0
        msg_target(mido.Message('note_off', note=int(note), channel=channel, time=delta_time))
    
    def note_on_score(self, note, velocity, voice, delta_time, msg_target):
        message = musicrep.NoteMessage(note, velocity, True, voice, delta_time)
        msg_target(message)
    
    def note_off_score(self, note, voice, delta_time, msg_target):
        message = musicrep.NoteMessage(note, 0, False, voice, delta_time)
        msg_target(message)
    
    def play(self, parent_base, wait, msg_target, realtime=True, delta_time=0, play_chords=True, play_drums=True, pitch_bends=None, free_channels=None):
        if pitch_bends is None:
            pitch_bends = {}
        if free_channels is None:
            free_channels = list(range(9))
        '''Play the mess of notes we have created, calling msg_target on each generated note'''
        if self.depth == 0: # Play leaf node
            # Play the starting chord
            if play_chords:
                current_chord = parent_base
                for note_i, note in enumerate(current_chord):
                    #msg_target(mido.Message('note_on', note = note + self.context.base_note, channel=0, velocity = 63 >> note_i, time = delta_time))
                    #self.note_on_midi(note + self.context.base_note, 63 >> note_i, pitch_bends, free_channels, delta_time, msg_target)
                    self.note_on_score(note + self.context.base_note, 63 >> note_i, 0, delta_time, msg_target)
                    delta_time = 0
            
            last_notes = set()
            for i in range(len(self.context.percussion[self.percussion])):
                if self.shift[i] is not None and play_chords: # Play the new chord
                    for note in current_chord:
                        #msg_target(mido.Message('note_off', note = note + self.context.base_note, channel=0, time = delta_time))
                        #self.note_off_midi(note + self.context.base_note, pitch_bends, free_channels, delta_time, msg_target)
                        self.note_off_score(note + self.context.base_note, 0, delta_time, msg_target)
                        delta_time = 0
                    current_chord = self.shift[i]
                    for note_i, note in enumerate(current_chord):
                        #msg_target(mido.Message('note_on', note = note + self.context.base_note, channel=0, velocity = 63 >> note_i, time = delta_time))
                        #self.note_on_midi(note + self.context.base_note, 63 >> note_i, pitch_bends, free_channels, delta_time, msg_target)
                        self.note_on_score(note + self.context.base_note, 63 >> note_i, 0, delta_time, msg_target)
                        delta_time = 0
                voices = self.children[i]
                new_notes = []
                for v, chord in enumerate(voices):
                    channel = self.context.voices[v].channel
                    note_offset = self.context.base_note + self.context.voices[v].base_note
                    for note, voice in last_notes: # Kill dead notes
                        if note not in chord and v == voice:
                            #msg_target(mido.Message('note_off', note = note + note_offset, channel = channel, time = delta_time))
                            #self.note_off_midi(note + note_offset, pitch_bends, free_channels, delta_time, msg_target)
                            self.note_off_score(note + note_offset, channel, delta_time, msg_target)
                            delta_time = 0
                    for note_i, note in enumerate(chord): # Play new notes
                        if (note, v) not in last_notes:
                            #msg_target(mido.Message('note_on', note = note + note_offset, channel = channel, velocity = 63 >> note_i, time = delta_time))
                            #self.note_on_midi(note + note_offset, 63 >> note_i, pitch_bends, free_channels, delta_time, msg_target)
                            self.note_on_score(note + note_offset, 63 >> note_i, channel, delta_time, msg_target)
                            delta_time = 0
                        new_notes.append((note, v))
                last_notes = new_notes
                
                drum = self.context.percussion[self.percussion][i]
                if not play_drums:
                    drum = 0
                if drum != 0: # Play drums if any
                    #msg_target(mido.Message('note_on', note = drum, channel = 9, velocity = 63, time = delta_time))
                    self.note_on_score(drum, 63, -1, delta_time, msg_target)
                    delta_time = 0
                    
                if realtime: # Wait if we're playing in real-time
                    time.sleep(wait)
                delta_time = int(delta_time + wait)
                
                if drum != 0: # Kill the drum if we played it
                    #msg_target(mido.Message('note_off', note = drum, channel = 9, time = delta_time))
                    self.note_off_score(drum, -1, delta_time, msg_target)
                    delta_time = 0
            
            if play_chords:
                for note in current_chord: # Kill remaining chord notes
                    #msg_target(mido.Message('note_off', note = note + self.context.base_note, channel=0, time = delta_time))
                    #self.note_off_midi(note + self.context.base_note, pitch_bends, free_channels, delta_time, msg_target)
                    self.note_off_score(note + self.context.base_note, 0, delta_time, msg_target)
                    delta_time = 0
            
            for note, voice in last_notes: # Kill remaining melody notes
                #msg_target(mido.Message(
                #    'note_off',
                #    note = note + self.context.base_note + self.context.voices[voice].base_note,
                #    channel = self.context.voices[voice].channel,
                #    time = delta_time))
                #self.note_off_midi(note + self.context.base_note, pitch_bends, free_channels, delta_time, msg_target)
                self.note_off_score(note + self.context.base_note, self.context.voices[voice].channel, delta_time, msg_target)
                delta_time = 0
        else: # Recursively play children nodes
            for child in self.children:
                delta_time = child.play(self.base, wait, msg_target, realtime=realtime, delta_time=delta_time, play_chords=play_chords, play_drums=play_drums, pitch_bends=pitch_bends, free_channels=free_channels)
        return delta_time
