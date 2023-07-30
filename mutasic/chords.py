'''MIDI note explorer'''

import argparse, random, time, math, typing, enum, sys, heapq
from dataclasses import dataclass, field
from os import path

import mido
import numpy as np

from mutasic import musicrep
from mutasic.generation import (
    ScaleModes,
    Note,
    create_scale,
    SongBuildingContext,
    all_chords,
    Measure,
    major_scale,
    minor_scale,
    harmonic_minor_scale,
    chromatic_scale,
    pent_major,
    pent_minor,
    blues_hex_major,
    blues_hex_minor,
    Voice
)

def play_drums(output, wait = 1):
    for i in range(35, 82):
        print(i)
        output.send(mido.Message('note_on', channel = 9, note = i, velocity = 63))
        time.sleep(wait)
        output.send(mido.Message('note_off', channel = 9, note = i))

def create_score(melody_scale_name='major',
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
    #mid = mido.MidiFile()
    #track = mido.MidiTrack()
    #mid.tracks.append(track)
    #track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    #mid.ticks_per_beat = tpb
    score = musicrep.Score(tpb, tempo)
    root.play(root.base, 1, lambda msg: score.append(msg), realtime = False, play_chords = play_chords, play_drums = play_drums)
    # track.append(mido.Message('note_off', note=0, channel=0, time=mid.ticks_per_beat))
    #return mid
    return score

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
        score = create_score(args.melody_scale,
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
        mid = score.to_midi(True)
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
    def __init__(self, score, microtonal=False):
        self.msgs = []
        self.microtonal = microtonal
        uniq_chans = set()
        time = 0
        score.convert(musicrep.ScoreTimeMode.RELATIVE)
        for msg in score.messages:
            time += msg.time
            self.msgs.append((time, msg))
            if msg.voice != -1:
                uniq_chans.add(msg.voice)
        self.length = time
        self.msgs.sort(key = lambda x: x[0])
        self.timescale = 1
        self.num_channels = len(uniq_chans)

@dataclass
class PatternState:
    '''A state in a pattern, with one bass and up to 6 melody components.'''
    bass: PatternPiece
    melodies: list[int] # 6 long
    
    def write(self, score, energy, melodies, base_channel=0):
        microtonal = any(map(lambda x: x.microtonal, melodies))
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
            channel = base_channel
            while index < len(self.bass.msgs):
                message = self.bass.msgs[index]
                if message[0] * self.bass.timescale > time:
                    break
                message = message[1].copy()
                message.time = delta_time
                if message.voice != 9:
                    message.voice = base_channel
                delta_time = 0
                score.append(message)
                index += 1
            channel += self.bass.num_channels
            for mel, nrg in mel_energy.items():
                melody = melodies[mel]
                while indices[mel] < len(melody.msgs):
                    message = melody.msgs[indices[mel]]
                    if message[0] * melody.timescale > time:
                        break
                    message = message[1].copy()
                    message.time = delta_time
                    if message.voice != 9:
                        message.voice = channels[mel] + base_channel
                    delta_time = 0
                    score.append(message)
                    if nrg >= 1:
                        message = message.copy()
                        message.time = delta_time
                        if message.frequency < 127 - 12:
                            message.frequency += 12 # 1 octave up
                        else:
                            message.frequency -= 12
                        score.append(message)
                    indices[mel] += 1
                channel += melody.num_channels
            time += 1
            delta_time += 1

def generate_pattern(state_length, num_states, transition_pow, basses, melodies, walk, tempo, bass_only, repeat, microtonal=False):
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
    #track = mido.MidiTrack()
    #track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    score = musicrep.Score(0, tempo)
    #for i in range(7):
    #    track.append(mido.Message('program_change', channel=i, program=i));
    for _, nrg in enumerate(energy):
        state = states[state_id]
        for __ in range(repeat):
            state.write(score, nrg, melodies)
        state_id = random.choices(range(num_states), weights=markov_chain[state_id,:])[0]
        
    return score

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
        basses.append(PatternPiece(create_score(args.melody_scale,
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
        melodies.append(PatternPiece(create_score(args.melody_scale,
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
    score = generate_pattern(args.length, args.states, args.power, basses, melodies, args.walk, args.tempo, args.intro_limit, args.repeat)
    score.ticks_per_beat = args.ticks_per_beat
    mid = score.to_midi(True)
    #mid = mido.MidiFile()
    #mid.tracks.append(track)
    #mid.ticks_per_beat = args.ticks_per_beat
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