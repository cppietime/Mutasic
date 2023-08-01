'''Representation that goes beyond midi'''

import enum
from dataclasses import dataclass, field, replace
import math
import struct
import sys
import wave

import mido
    
def _to_vle(value):
    buf = bytearray()
    last_byte = 0
    while value or not last_byte:
        buf.append((value & 0x7F) | last_byte)
        last_byte = 0x80
        value >>= 7
    return bytes(buf[::-1])

def _from_vle(src):
    value = 0
    while True:
        b = src.read(1)[0]
        value <<= 7
        value |= b & 0x7F
        if not (value & 0x80):
            break
    return value

def _midi_to_hz(note):
    return 440 * 2 ** ((note - 69) / 12)

class ScoreTimeMode(enum.Enum):
    ABSOLUTE = enum.auto()
    RELATIVE = enum.auto()
    
@dataclass
class NoteMessage:
    frequency: float
    amplitude: int
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
    voice_range: list[int] = field(default_factory=lambda: [None, None])
    max_time: int = 0
    
    def append(self, message):
        self.messages.append(message)
        if self.time_mode == ScoreTimeMode.RELATIVE:
            self.max_time += message.time
        else:
            self.max_time = max(self.max_time, message.time)
        if message.voice != -1:
            if self.voice_range[0] is None or message.voice < self.voice_range[0]:
                self.voice_range[0] = message.voice
            if self.voice_range[1] is None or message.voice > self.voice_range[1]:
                self.voice_range[1] = message.voice
    
    def check_voice_range(self):
        for message in self.messages:
            if message.voice != -1:
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
    
    def save(self, dest=sys.stdout.buffer, binary=True):
        self.convert(ScoreTimeMode.RELATIVE)
        self.check_voice_range()
        if isinstance(dest, str):
            with open(dest, 'wb') as file:
                if binary:
                    self._save_binary(file)
                else:
                    self._save_ascii(file)
        else:
            if binary:
                self._save_binary(dest)
            else:
                self._save_ascii(dest)
    
    def _save_binary(self, dest):
        header = struct.pack('<4sIIHBBfHHII', b'dscB', 0, self.usec_per_beat, self.ticks_per_beat, 69, 0, 440, *self.voice_range, self.max_time, len(self.messages))
        dest.write(header)
        for msg in self.messages:
            dest.write(_to_vle(msg.time))
            amp = msg.amplitude if msg.begins else 0
            note_no = int(msg.frequency)
            octlets = int((msg.frequency - note_no) * 256)
            note_head = struct.pack('<HBB', msg.voice & 0xffff, amp, note_no)
            if octlets:
                note_head += struct.pack('<B', octlets)
            note_head += _to_vle(0)
            dest.write(note_head)
    
    def _save_ascii(self, dest):
        header = f'dscA {self.usec_per_beat} {self.ticks_per_beat}\n\
{self.voice_range[0]} {self.voice_range[1]}\n\
{self.max_time} {len(self.messages)}\n'
        dest.write(header.encode())
        for msg in self.messages:
            amp = msg.amplitude if msg.begins else 0
            smsg = f'{msg.time} {amp} {msg.frequency} {msg.voice & 0xffff}\n'
            dest.write(smsg.encode())
    
    def to_wav(self, dest):
        self.convert(ScoreTimeMode.RELATIVE)
        with wave.open(dest, 'wb') as wav:
            wav.setframerate(44100)
            wav.setsampwidth(2)
            wav.setnchannels(1)
            wav.setcomptype('NONE', 'not compressed')
            data = bytearray()
            playing_notes = {} # (pitch, voice): [amp, phase]
            usecs = 0
            for msg in self.messages:
                print(msg)
                if msg.time:
                    usecs += msg.time / self.ticks_per_beat * self.usec_per_beat # usec
                    nsamps = int(usecs * 44100 * 1e-6)
                    usecs -= nsamps * 1e6 / 44100
                    
                    for _ in range(nsamps):
                        sample = 0
                        for pv, ap in playing_notes.items():
                            pitch, _ = pv
                            amp, phase = ap
                            #sample += ((phase % (2 * math.pi)) / (2 * math.pi) * 2 - 1) * amp / 127 / 4
                            #sample += (1 if phase % (2 * math.pi) <= math.pi else -1) * amp / 127 / 4
                            sample += (lambda p: min(p, 2 - p) * 2 - 1)(phase % (2 * math.pi) / (math.pi)) * amp / 127 / 4
                            #sample += math.sin(phase) * amp / 127
                            phase += math.pi * 2 * pitch / 44100
                            ap[1] = phase
                        sample = min(max(-32768, sample * 32767 / 6), 32767)
                        short = int(sample) & 0xffff
                        data += short.to_bytes(2, 'little')
                    
                freq = _midi_to_hz(msg.frequency)
                key = (freq, msg.voice)
                if msg.amplitude == 0:
                    playing_notes.pop(key)
                else:
                    playing_notes[key] = [msg.amplitude, 0]
            wav.setnframes(len(data) // 2)
            wav.writeframes(data)
    
    @staticmethod
    def load(src):
        if isinstance(src, str):
            with open(src, 'rb') as file:
                return Score._load(file)
        return Score._load(src)
    
    @staticmethod
    def _load(src):
        fourcc = src.read(4)
        if fourcc == b'dscA':
            return Score._load_ascii(src)
        elif fourcc == b'dscB':
            return Score._load_binary(src)
        raise ValueError('Provided file is not a valid digital score.')
    
    @staticmethod
    def _load_ascii(src):
        buffer = bytearray()
        def skip_ws():
            while True:
                b = src.read(1)
                if b not in {b'\n', b' ', b'\t'}:
                    buffer.extend(b)
                    return
                if not b:
                    return
        def read_token():
            buffer.clear()
            skip_ws()
            while True:
                b = src.read(1)
                if b in {b'\n', b' ', b'\t'} or not b:
                    break;
                buffer.extend(b)
            token = bytes(buffer).decode()
            buffer.clear()
            return token
        usec_per_beat = int(read_token())
        ticks_per_beat = int(read_token())
        voice_min = int(read_token())
        voice_max = int(read_token())
        time_max = int(read_token())
        num_msgs = int(read_token())
        msgs = []
        for _ in range(num_msgs):
            time = int(read_token())
            amp = int(read_token())
            pitch = float(read_token())
            voice = int(read_token())
            msgs.append(NoteMessage(pitch, amp, amp != 0, voice, time))
        return Score(ticks_per_beat, usec_per_beat, msgs, ScoreTimeMode.RELATIVE, [voice_min, voice_max], time_max)
    
    @staticmethod
    def _load_binary(src):
        filesize, usec_per_beat, ticks_per_beat, ref_note, ref_octlets, ref_freq, voice_min, voice_max, time_max, num_msgs = struct.unpack('<IIHBBfHHII', src.read(28))
        msgs = []
        for _ in range(num_msgs):
            time = _from_vle(src)
            voice, amp, note = struct.unpack('<HBB', src.read(4))
            octlets = 0
            if note & 0x80:
                octlets = src.read(1)[0]
                note &= 0x7F
            num_extra = _from_vle(src)
            extras = src.read(num_extra)
            pitch = note + octlets / 256
            msgs.append(NoteMessage(pitch, amp, amp != 0, voice, time))
        return Score(ticks_per_beat, usec_per_beat, msgs, ScoreTimeMode.RELATIVE, [voice_min, voice_max], time_max)
