import io
import random

import mido

def microtonal():
    mid = mido.MidiFile()
    mid.ticks_per_beat = 8
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set all note tunings to midi 60
    notes = []
    for i in range(127):
        notes += [i, 60, 0, 0]
    track.append(mido.Message('sysex', data=(0x7F, 0x7F, 8, 2, 0, 127) + tuple(notes)))
    
    # Also try sending a bulk tuning dump?
    notes = []
    for i in range(128):
        notes += [60, 0, 0]
    track.append(mido.Message('sysex', data=(0x7E, 0x7F, 8, 1, 0) + tuple(b'0' * 16) + tuple(notes) + (0,)))
    
    # Set tuning program to 0
    track.append(mido.Message('control_change', control=64, value=3))
    track.append(mido.Message('control_change', control=65, value=0))
    track.append(mido.Message('control_change', control=6, value=0))
    
    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=2_000_000))
    
    # Some notes
    for note in (40, 41, 42, 43, 60, 59):
        track.append(mido.Message('note_on', channel=0, velocity=63, note=note, time=0))
        track.append(mido.Message('note_off', channel=0, velocity=63, note=note, time=2))
    
    buffer = io.BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)
    value = buffer.read()
    with open('micro.mid', 'wb') as file:
        file.write(value)
    
    output = mido.open_output()
    for msg in mid.play():
        output.send(msg)
    output.close()

def multitrack():
    mid = mido.MidiFile()
    mid.ticks_per_beat = 8
    trackt = mido.MidiTrack()
    mid.tracks.append(trackt)
    track0 = mido.MidiTrack()
    mid.tracks.append(track0)
    track1 = mido.MidiTrack()
    mid.tracks.append(track1)
    
    trackt.append(mido.MetaMessage('set_tempo', tempo=2_000_000))
    track0.append(mido.Message('program_change', channel=0, program=1))
    track1.append(mido.Message('program_change', channel=0, program=120))
    
    for note in (40, 42, 44, 45):
        track0.append(mido.Message('note_on', channel=0, velocity=63, note=note, time=0))
        track0.append(mido.Message('note_off', channel=0, velocity=63, note=note, time=1))
        track1.append(mido.Message('note_on', channel=0, velocity=63, note=note, time=0))
        track1.append(mido.Message('note_off', channel=0, velocity=63, note=note, time=1))
    
    mid.save('tracks.mid')

if __name__ == '__main__':
    microtonal()