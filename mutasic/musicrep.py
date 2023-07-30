'''Representation that goes beyond midi'''

import enum
from dataclasses import dataclass, field, replace

import mido

class ScoreTimeMode(enum.Enum):
    ABSOLUTE = enum.auto()
    RELATIVE = enum.auto()
    
@dataclass
class NoteMessage:
    frequency: float
    amplitude: float
    begins: bool
    voice: int
    time: int = 0
    
    def copy(self):
        return replace(self)

@dataclass
class Score:
    ticks_per_beat: int
    usec_per_beat: int
    messages: list[NoteMessage] = field(default_factory=list)
    time_mode: ScoreTimeMode = ScoreTimeMode.RELATIVE
    voice_range: list[int] = field(init=False, default_factory=lambda: [None, None])
    
    def append(self, message):
        self.messages.append(message)
        if self.voice_range[0] is None or message.voice < self.voice_range[0]:
            self.voice_range[0] = message.voice
        if self.voice_range[1] is None or message.voice > self.voice_range[1]:
            self.voice_range[1] = message.voice
    
    def convert(self, time_mode):
        '''Converts between relative and absolute time modes, in place.'''
        if time_mode == self.time_mode:
            return
        if time_mode == ScoreTimeMode.ABSOLUTE:
            # Convert relative to absolute
            time = 0
            for message in self.messages:
                delta_time = message.time
                time += delta_time
                message.time = time
        else:
            # Convert absolute to relative
            sort(self.messages, key=lambda x: x.time)
            time = 0
            for message in self.messages:
                delta_time = message.time - time
                message.time = delta_time
                time += delta_time
    
    def to_midi(self, include_tempo=False):
        track = mido.MidiTrack()
        if include_tempo:
            track.append(mido.MetaMessage('set_tempo', tempo=self.usec_per_beat))
        self.convert(ScoreTimeMode.RELATIVE)
        for message in self.messages:
            channel = 9 if message.voice == -1 else message.voice
            midimsg = mido.Message('note_on' if message.begins else 'note_off', note=int(message.frequency), velocity=int(message.amplitude), channel=channel, time=message.time)
            track.append(midimsg)
        mid = mido.MidiFile()
        mid.ticks_per_beat = self.ticks_per_beat
        mid.tracks.append(track)
        return mid
