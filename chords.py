'''MIDI note explorer'''

import argparse, random, time, math, typing, enum, sys, heapq
from dataclasses import dataclass, field
from os import path

import mido
import numpy as np

# TODO allow microtonality

major_scale = [0, 2, 4, 5, 7, 9, 11]
minor_scale = [0, 2, 3, 5, 7, 8, 10]
harmonic_minor_scale = [0, 2, 3, 5, 7, 8, 11]
chromatic_scale = tuple(range(12))
pent_major = [0, 2, 4, 7, 9]
pent_minor = [0, 3, 5, 7, 10]
blues_hex_major = [0, 2, 3, 4, 7, 9]
blues_hex_minor = [0, 3, 5, 6, 7, 10]
#blues_hex_minor = [0, 0.25, 0.75, 1.33, 1.75]
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
    
    def note_on(self, note, velocity, pitch_bends, free_channels, delta_time, msg_target):
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
    
    def note_off(self, note, pitch_bends, free_channels, delta_time, msg_target):
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
                    self.note_on(note + self.context.base_note, 63 >> note_i, pitch_bends, free_channels, delta_time, msg_target)
                    delta_time = 0
            
            last_notes = set()
            for i in range(len(self.context.percussion[self.percussion])):
                if self.shift[i] is not None and play_chords: # Play the new chord
                    for note in current_chord:
                        #msg_target(mido.Message('note_off', note = note + self.context.base_note, channel=0, time = delta_time))
                        self.note_off(note + self.context.base_note, pitch_bends, free_channels, delta_time, msg_target)
                        delta_time = 0
                    current_chord = self.shift[i]
                    for note_i, note in enumerate(current_chord):
                        #msg_target(mido.Message('note_on', note = note + self.context.base_note, channel=0, velocity = 63 >> note_i, time = delta_time))
                        self.note_on(note + self.context.base_note, 63 >> note_i, pitch_bends, free_channels, delta_time, msg_target)
                        delta_time = 0
                voices = self.children[i]
                new_notes = []
                for v, chord in enumerate(voices):
                    channel = self.context.voices[v].channel
                    note_offset = self.context.base_note + self.context.voices[v].base_note
                    for note, voice in last_notes: # Kill dead notes
                        if note not in chord and v == voice:
                            #msg_target(mido.Message('note_off', note = note + note_offset, channel = channel, time = delta_time))
                            self.note_off(note + note_offset, pitch_bends, free_channels, delta_time, msg_target)
                            delta_time = 0
                    for note_i, note in enumerate(chord): # Play new notes
                        if (note, v) not in last_notes:
                            #msg_target(mido.Message('note_on', note = note + note_offset, channel = channel, velocity = 63 >> note_i, time = delta_time))
                            self.note_on(note + note_offset, 63 >> note_i, pitch_bends, free_channels, delta_time, msg_target)
                            delta_time = 0
                        new_notes.append((note, v))
                last_notes = new_notes
                
                drum = self.context.percussion[self.percussion][i]
                if not play_drums:
                    drum = 0
                if drum != 0: # Play drums if any
                    msg_target(mido.Message('note_on', note = drum, channel = 9, velocity = 63, time = delta_time))
                    delta_time = 0
                    
                if realtime: # Wait if we're playing in real-time
                    time.sleep(wait)
                delta_time = int(delta_time + wait)
                
                if drum != 0: # Kill the drum if we played it
                    msg_target(mido.Message('note_off', note = drum, channel = 9, time = delta_time))
                    delta_time = 0
            
            if play_chords:
                for note in current_chord: # Kill remaining chord notes
                    #msg_target(mido.Message('note_off', note = note + self.context.base_note, channel=0, time = delta_time))
                    self.note_off(note + self.context.base_note, pitch_bends, free_channels, delta_time, msg_target)
                    delta_time = 0
            
            for note, voice in last_notes: # Kill remaining melody notes
                #msg_target(mido.Message(
                #    'note_off',
                #    note = note + self.context.base_note + self.context.voices[voice].base_note,
                #    channel = self.context.voices[voice].channel,
                #    time = delta_time))
                self.note_off(note + self.context.base_note, pitch_bends, free_channels, delta_time, msg_target)
                delta_time = 0
        else: # Recursively play children nodes
            for child in self.children:
                delta_time = child.play(self.base, wait, msg_target, realtime=realtime, delta_time=delta_time, play_chords=play_chords, play_drums=play_drums, pitch_bends=pitch_bends, free_channels=free_channels)
        return delta_time

def play_drums(output, wait = 1):
    for i in range(35, 82):
        print(i)
        output.send(mido.Message('note_on', channel = 9, note = i, velocity = 63))
        time.sleep(wait)
        output.send(mido.Message('note_off', channel = 9, note = i))

def create_midi(melody_scale_name='major',
                melody_scale_mode='IONIAN',
                melody_starting_note='C',
                num_melody_octaves=2,
                chord_length_min=2,
                chord_length_max=3,
                chord_scale_name='major',
                jazz_on_beat=.1,
                jazz_off_beat=.1,
                melody_base=42,
                mutation=.8,
                chord_instability=.125,
                tempo=1_000_000,
                tpb=8,
                depth=1,
                voices=1,
                play_chords=True,
                play_drums=True):
    scales = {
        'major': major_scale,
        'minor': minor_scale,
        'harmonic': harmonic_minor_scale,
        'chromatic': chromatic_scale,
        'major_pent': pent_major,
        'minor_pent': pent_minor,
        'major_blues': blues_hex_major,
        'minor_blues': blues_hex_minor
    }
    assert chord_length_max >= chord_length_min
    scale_base = scales[melody_scale_name.lower()]
    mode = ScaleModes[melody_scale_mode.upper()]
    starting_note = Note[melody_starting_note.upper()]
    print(scale_base, mode, starting_note)
    scale = create_scale(scale=scale_base, octaves=num_melody_octaves, mode=mode, base_note=starting_note)
    chord_scale = create_scale(scale=scales[chord_scale_name.lower()], octaves=1, mode=mode, base_note=starting_note)
    context = SongBuildingContext(drum_beat_offset=0,
        notes=scale,
        chords=all_chords((chord_length_min, chord_length_max), out_of=chord_scale),
        jazz=(jazz_on_beat, jazz_off_beat),
        base_note=melody_base,
        base_mutation_rate=mutation,
        chord_instability=chord_instability,
        voices=[Voice(i+1, 1, 0) for i in range(voices)])
    root = Measure(depth, [0], context)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    mid.ticks_per_beat = tpb
    root.play(root.base, 1, lambda msg: track.append(msg), realtime = False, play_chords = play_chords, play_drums = play_drums)
    # track.append(mido.Message('note_off', note=0, channel=0, time=mid.ticks_per_beat))
    return mid

def play_rt(midi):
    port = mido.open_output()
    for msg in midi.play():
        port.send(msg)
    port.close()

def create_cmd():
    parser = argparse.ArgumentParser('Create a MIDI song')
    parser.add_argument('--melody_scale', '-s', type=str, default='major', help='Name of scale')
    parser.add_argument('--octaves', type=int, default=2)
    parser.add_argument('--mode', type=str, default='IONIAN')
    parser.add_argument('--scale_base', type=str, default='C')
    parser.add_argument('--min_chord', type=int, default=2)
    parser.add_argument('--max_chord', type=int, default=3)
    parser.add_argument('--chord_scale', type=str, default=None)
    parser.add_argument('--jazz_on', type=float, default=.1)
    parser.add_argument('--jazz_off', type=float, default=.2)
    parser.add_argument('--base_note', type=int, default=42)
    parser.add_argument('--mutation', type=float, default=.8)
    parser.add_argument('--instability', type=float, default=.125)
    parser.add_argument('--tempo', type=int, default=1_000_000)
    parser.add_argument('--ticks_per_beat', type=int, default=8)
    parser.add_argument('--depth', '-n', type=int, default=1)
    parser.add_argument('--seed', '-x', type=int, default=None)
    parser.add_argument('--voices', type=int, default=1)
    parser.add_argument('--omit_chords', action='store_true')
    parser.add_argument('--output', type=str)
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--drums', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    if not args.chord_scale:
        args.chord_scale = args.melody_scale
    if not args.seed:
        args.seed = random.getrandbits(64)
    print(f'seed={args.seed}', file=sys.stderr)
    random.seed(args.seed)
    for i in range(args.times):
        mid = create_midi(args.melody_scale,
                          args.mode,
                          args.scale_base,
                          args.octaves,
                          args.min_chord,
                          args.max_chord,
                          args.chord_scale,
                          args.jazz_on,
                          args.jazz_off,
                          args.base_note,
                          args.mutation,
                          args.instability,
                          args.tempo,
                          args.ticks_per_beat,
                          args.depth,
                          args.voices,
                          not args.omit_chords,
                          args.drums)
        if args.output:
            output = args.output
            if args.times > 1:
                output = path.splitext(output)
                base = f'{output[0]}_{i:05}'
                output = base + output[1]
            mid.save(output)
        else:
            play_rt(mid)
            output = input('Name of saved midi file? [blank to discard] > ')
            if output:
                mid.save(output)

class PatternPiece:
    '''One melodic/bass piece of a pattern. Stacked up to make states.
    Upon initialization, all tracks are merged together.
    '''
    def __init__(self, midifile):
        self.length = 0
        self.msgs = []
        for track in midifile.tracks:
            time = 0
            for msg in track:
                time += msg.time
                if msg.type in {'note_on', 'note_off'}:
                    self.msgs.append((time, msg))
            self.length = max(self.length, time)
        self.msgs.sort(key = lambda x: x[0])
        self.timescale = 1

@dataclass
class PatternState:
    '''A state in a pattern, with one bass and up to 6 melody components.'''
    bass: PatternPiece
    melodies: list[int] # 6 long
    
    def write(self, track, energy, melodies):
        mel_energy = {}
        channels = {}
        for i in range(energy):
            mel_energy[self.melodies[i]] = mel_energy.get(self.melodies[i], -1) + 1
            channels.setdefault(self.melodies[i], len(channels) + 1)
        print(f'{energy=} {self.melodies=} {mel_energy=}')
        time = 0 # Bass
        index = 0 # Bass
        delta_time = 0
        times = {i: 0 for i in mel_energy}
        indices = {i: 0 for i in mel_energy}
        while time < self.bass.length * self.bass.timescale:
            while index < len(self.bass.msgs):
                message = self.bass.msgs[index]
                if message[0] * self.bass.timescale > time:
                    break
                message = message[1].copy()
                message.time = delta_time
                if message.type in ('note_on', 'note_off') and message.channel != 9:
                    message.channel = 0
                delta_time = 0
                track.append(message)
                index += 1
            for mel, nrg in mel_energy.items():
                melody = melodies[mel]
                while indices[mel] < len(melody.msgs):
                    message = melody.msgs[indices[mel]]
                    if message[0] * melody.timescale > time:
                        break
                    message = message[1].copy()
                    message.time = delta_time
                    if message.type in ('note_on', 'note_off') and message.channel != 9:
                        message.channel = channels[mel]
                    delta_time = 0
                    track.append(message)
                    if message.type in ('note_on', 'note_off') and nrg >= 1:
                        message = message.copy()
                        message.time = delta_time
                        if message.note < 127 - 12:
                            message.note += 12 # 1 octave up
                        else:
                            message.note -= 12
                        track.append(message)
                    indices[mel] += 1
            time += 1
            delta_time += 1

def generate_pattern(state_length, num_states, transition_pow, basses, melodies, walk, tempo, bass_only, repeat):
    '''Generate a MIDI track pattern. Each pattern piece gets its own channel.
    
    state_length: length in pattern in number of states, before repetition
    num_states: number of unique states to generate
    transition_pow: power for elements of markov matrix
    basses: bass pieces
    melodies: melody pieces
    walk: random walk distance limits
    tempo: MIDI tempo
    bass_only: max number of intro states with only bass
    repeat: number of times to play each state
    '''
    # Energy outline
    energy = np.zeros(state_length, dtype=int)
    last_energy = 0
    for i in range(1, state_length):
        diff = random.randint(-walk, walk)
        energy[i] = energy[i - 1] + diff
        min_energy = 0 if bass_only < 1 or i < bass_only else 1
        energy[i] = np.clip(energy[i], min_energy, 6) # Assume max bassline + 3 high energy melodies
    energy[state_length-2] = min(energy[state_length-2], 3)
    energy[state_length-1] = min(energy[state_length-1], 1)
    print(f'{energy=}')
    
    # State transitions
    markov_chain = np.random.random(size=(num_states, num_states))
    for from_ in range(num_states):
        state = markov_chain[from_,:]
        state /= state.sum()
        state **= transition_pow
        state /= state.sum()
    
    # Set timescales
    #basses = tuple(map(PatternPiece, basses))
    #melodies = tuple(map(PatternPiece, melodies))
    max_len = 0
    for piece in tuple(basses) + tuple(melodies):
        max_len = max(max_len, piece.length)
    for piece in tuple(basses) + tuple(melodies):
        if max_len % piece.length != 0:
            print("Pieces' lengths indivisible", file=sys.stderr)
            sys.exit(1)
        piece.timescale = max_len // piece.length
        print(f'{piece.timescale=} {piece.length=}')
    
    states = []
    for _ in range(state_length):
        bass = random.choice(basses)
        if len(melodies) < 3:
            mel_ids = random.choices(range(len(melodies)), k=6)
        else:
            mel_ids = random.sample(range(len(melodies)), 3) * 2
            random.shuffle(mel_ids)
        states.append(PatternState(bass, mel_ids))
    
    state_id = 0
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    notes = []
    for i in range(127):
        notes += [i, 60, 63, 0]
    track.append(mido.Message('sysex', data=(0x7F, 0, 8, 2, 0, 127) + tuple(notes)))
    track.append(mido.Message('control_change', control=64, value=3))
    track.append(mido.Message('control_change', control=65, value=0))
    track.append(mido.Message('control_change', control=6, value=0))
    for i in range(7):
        track.append(mido.Message('program_change', channel=i, program=i));
    for _, nrg in enumerate(energy):
        state = states[state_id]
        for __ in range(repeat):
            state.write(track, nrg, melodies)
        state_id = random.choices(range(num_states), weights=markov_chain[state_id,:])[0]
    
    return track

def pattern_cmd():
    bass_names = sys.argv[2].split(',')
    melody_names = sys.argv[3].split(',')
    length = int(sys.argv[4])
    states = int(sys.argv[5])
    power = 1
    
    basses = []
    for bass_name in bass_names:
        mid = mido.MidiFile(bass_name)
        basses.append(PatternPiece(mid))
    melodies = []
    for melody_name in melody_names:
        mid = mido.MidiFile(melody_name)
        melodies.append(PatternPiece(mid))
    
    track = generate_pattern(length, states, power, basses, melodies)
    midi = mido.MidiFile()
    midi.ticks_per_beat = 8
    midi.tracks.append(track)
    play_rt(midi)
    midi.save('pattern.mid')

def auto_cmd():
    parser = argparse.ArgumentParser('Create a MIDI song')
    parser.add_argument('--melody_scale', '-s', type=str, default='major', help='Name of scale')
    parser.add_argument('--octaves', type=int, default=2)
    parser.add_argument('--mode', type=str, default='IONIAN')
    parser.add_argument('--scale_base', type=str, default='C')
    parser.add_argument('--min_chord', type=int, default=2)
    parser.add_argument('--max_chord', type=int, default=3)
    parser.add_argument('--chord_scale', type=str, default=None)
    parser.add_argument('--jazz_on', type=float, default=.1)
    parser.add_argument('--jazz_off', type=float, default=.2)
    parser.add_argument('--base_note', type=int, default=42)
    parser.add_argument('--mutation', type=float, default=.8)
    parser.add_argument('--instability', type=float, default=.125)
    parser.add_argument('--tempo', type=int, default=1_000_000)
    parser.add_argument('--ticks_per_beat', type=int, default=8)
    parser.add_argument('--depth', '-n', type=int, default=1)
    parser.add_argument('--seed', '-x', type=int, default=None)
    parser.add_argument('--voices', type=int, default=1)
    parser.add_argument('--omit_chords', action='store_true')
    parser.add_argument('--output', type=str)
    parser.add_argument('--basses', type=int, default=1)
    parser.add_argument('--melodies', type=int, default=1)
    parser.add_argument('--states', type=int, default=1)
    parser.add_argument('--length', type=int, default=1)
    parser.add_argument('--power', type=float, default=1)
    parser.add_argument('--drums', action='store_true')
    parser.add_argument('--walk', type=int, default=2)
    parser.add_argument('--octave_distance', type=int, default=1)
    parser.add_argument('--intro_limit', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args(sys.argv[2:])
    if not args.chord_scale:
        args.chord_scale = args.melody_scale
    if not args.seed:
        args.seed = random.getrandbits(64)
    print(f'seed={args.seed}', file=sys.stderr)
    random.seed(args.seed)
    basses, melodies = [], []
    for _ in range(args.basses):
        basses.append(PatternPiece(create_midi(args.melody_scale,
                          args.mode,
                          args.scale_base,
                          args.octaves,
                          args.min_chord,
                          args.max_chord,
                          args.chord_scale,
                          args.jazz_on,
                          args.jazz_off,
                          args.base_note,
                          args.mutation,
                          args.instability,
                          args.tempo,
                          args.ticks_per_beat,
                          args.depth,
                          args.voices,
                          not args.omit_chords,
                          args.drums)))
    for _ in range(args.melodies):
        melodies.append(PatternPiece(create_midi(args.melody_scale,
                          args.mode,
                          args.scale_base,
                          args.octaves,
                          args.min_chord,
                          args.max_chord,
                          args.chord_scale,
                          args.jazz_on,
                          args.jazz_off,
                          args.base_note + args.octave_distance * 12,
                          args.mutation,
                          args.instability,
                          args.tempo,
                          args.ticks_per_beat,
                          args.depth,
                          args.voices,
                          not args.omit_chords,
                          False)))
    track = generate_pattern(args.length, args.states, args.power, basses, melodies, args.walk, args.tempo, args.intro_limit, args.repeat)
    mid = mido.MidiFile()
    mid.tracks.append(track)
    mid.ticks_per_beat = args.ticks_per_beat
    if args.output:
        mid.save(args.output)
    else:
        play_rt(mid)
        output = input('Enter filename to save [blank to discard]: ')
        if output:
            mid.save(output)

def main():
    if len(sys.argv) < 2:
        print('Must specify a command', file=sys.stderr)
        sys.exit(1)
    command = sys.argv[1]
    if command == 'create':
        create_cmd()
    elif command == 'pattern':
        pattern_cmd()
    elif command == 'compose':
        auto_cmd()
    else:
        print(f'Unspecified command {command}')
        sys.exit(1)

if __name__ == '__main__':
    main()