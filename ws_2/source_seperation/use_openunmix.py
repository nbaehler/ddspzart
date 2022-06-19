from pyparsing import Optional
import torch
import random
import mirdata
import os
from benedict import benedict
import numpy as np
import logging
import time

import yaml
import pickle

import tqdm
import musdb

DATASET_FOLDER = "/media/olaf/OlafSSD/03_Dokumente/01_epfl/01_cm/slakh2100_flac_redux"

def get_sequence_from_start_and_duration(x: np.ndarray, seq_duration: float, start_time: float, sr: float) -> np.ndarray:
    start_idx = int(start_time * sr)
    nb_samples_seq = int(seq_duration * sr)
    audio_snippet = x[start_idx: start_idx + nb_samples_seq]
    return audio_snippet

class Target():

    def __init__(self, track_id: int, stem_id: int, split: str = 'train',seq_duration: float = 5.) -> None:
        self.track_id, self.stem_id = track_id, stem_id

        self.seq_duration = seq_duration

        stem = f"S{stem_id:02d}"
        track_folder = f"{DATASET_FOLDER}/{split}/Track{track_id:05d}"
        self.audio_stem_folder = f"{track_folder}/stems/{stem}.flac"
        # tic = time.perf_counter()
        # with open(f"{track_folder}/metadata.yaml", 'r') as stream:
        #     try:
        #         info_dict=yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)
        info_dict = benedict.from_yaml( f"{track_folder}/metadata.yaml")
        # toc = time.perf_counter()
        # print("bene", toc-tic)
        self.instrument = info_dict["stems"][stem]["midi_program_name"]
        self.instrument_cls = info_dict["stems"][stem]["inst_class"]
        self.sample_rate = None
    
    def get_audio_seq(self, start_time: float = None) -> np.ndarray:
        # tic = time.perf_counter()
        x, sr = mirdata.datasets.slakh.load_audio(self.audio_stem_folder)
        # toc = time.perf_counter()
        # print("get stem", toc - tic)
        self.sample_rate = sr
        if start_time is None:
            duration = len(x) / sr
            start_time = random.uniform(0, duration - self.seq_duration)
        # print(start_time)
        return get_sequence_from_start_and_duration(x,self.seq_duration,start_time,sr)

    def get_total_audio(self) -> np.ndarray:
        x, sr = mirdata.datasets.slakh.load_audio(self.audio_stem_folder)
        return x

    def __repr__(self) -> str:
        return f"Target with track {self.track_id}, stem {self.stem_id} and instrument: {self.instrument}."

class Track():

    def __init__(self, track_id: int, split: str = 'train', seq_duration: float = 5.) -> None:
        self.track_folder = f"{DATASET_FOLDER}/{split}/Track{track_id:05d}"
        self.track_id = track_id
        self.split = split
        self.audio_mix_folder = f"{self.track_folder}/mix.flac"
        self.seq_duration = seq_duration
        self.start_time = None
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_targets_file = f"{file_dir}/data/targets/{split}/save_targets_{track_id:05d}.pickle"

        self.get_nb_audio_samples()
        self.init_targets()
    
    def init_targets(self):
        stem_ids = [int(stem[-7:-5]) for stem in os.listdir(f"{self.track_folder}/stems")]
        # tic = time.perf_counter()
        if os.path.exists(self.save_targets_file) and self.split == "train":
            with open(self.save_targets_file, 'rb') as targets_file:
                self.targets = pickle.load(targets_file)
            # logging.info("Loaded targets from pickle file.")
        else:
            self.targets = [Target(self.track_id, stem_id, split=self.split, seq_duration=self.seq_duration) for stem_id in stem_ids]
            with open(self.save_targets_file, 'wb') as pickle_file:
                pickle.dump(self.targets, pickle_file)
            # logging.info("Initialized targets and loaded to pickle file.")
        # toc = time.perf_counter()
        # print("init target", toc-tic)

    def __repr__(self) -> str:
        return f"Track {self.track_id}" # with {[t.instrument for t in self.targets]}."

    def get_mixed_audio(self) -> np.ndarray:
        # tic = time.perf_counter()
        x, sr = mirdata.datasets.slakh.load_audio(self.audio_mix_folder)
        # toc = time.perf_counter()
        # print("get mix", toc-tic)
        start_time = random.uniform(0, self.duration - self.seq_duration)
        self.start_time = start_time
        #self.start_time = 50 #FIXME REmove
        return get_sequence_from_start_and_duration(x,self.seq_duration,self.start_time,sr)

    def get_nb_audio_samples(self):
        x, sr = mirdata.datasets.slakh.load_audio(self.audio_mix_folder)
        self.sample_rate = sr
        self.nb_audio_samples = len(x)
        self.duration = len(x) / sr

    def get_stem_audio(self) -> np.ndarray:
        xs = np.zeros((self.nb_audio_samples, len(self.targets)))
        for i,target in enumerate(self.targets):
            xs[:,i] = target.get_audio_seq()
        return xs
    
    def get_all_audio_seqs(self):
        mix_audio = self.get_mixed_audio()
        stem_audio = np.zeros((int(self.seq_duration*self.sample_rate), len(self.targets)))
        for i,target in enumerate(self.targets):
            stem_audio[:,i] = target.get_audio_seq(start_time=self.start_time)
        return mix_audio, stem_audio

    def get_selected_audio_seqs(self, target: Target):
        # tic = time.perf_counter()
        mix_audio = self.get_mixed_audio()
        # toc = time.perf_counter()
        stem_audio = target.get_audio_seq(self.start_time)

        if len(stem_audio) is 0 or (len(stem_audio) != self.seq_duration * self.sample_rate):
            logging.info("Set stem_audio to zero.")
            stem_audio = np.zeros_like(mix_audio)
        # toc2 = time.perf_counter()
        # print("get sel", toc-tic, toc2-toc)
        return mix_audio, stem_audio


class SlakhDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split='train',
        target='French Horn',
        seq_duration=5.,
    ):
        """Slakh Dataset wrapper
        """
        self.target = target
        self.dataset = mirdata.initialize('slakh',DATASET_FOLDER)
        track_folders = sorted(os.listdir(f"{DATASET_FOLDER}/{split}"))[:10]
        self.all_tracks = []
        # for id_str in tqdm.notebook.tqdm(track_folders):
        # for id_str in tqdm.tqdm(track_folders):
            # self.all_tracks.append(Track(int(id_str[-5:]), split=split, seq_duration=seq_duration))
        self.all_tracks = [Track(int(id_str[-5:]), split=split, seq_duration=seq_duration) for id_str in track_folders]
        self.filter_target()

    def filter_target(self):
        des_targets = list()
        des_tracks = list()
        for tr,track in enumerate(self.all_tracks):
            cnt = 0
            for ta, target in enumerate(track.targets):
                if target.instrument == self.target:
                    if cnt >= 0: # not used currently
                        des_targets.append(target)
                        des_tracks.append(track)
                    else:
                        logging.info("Do not add.") # I just add all elements...
                    cnt += 1
        self.tracks = des_tracks
        self.targets = des_targets
        if len(self.tracks) != len(self.targets):
            logging.warning("At least two targets are the same in this track.")

        
    def __getitem__(self, index: int):
        # tic = time.perf_counter()
        track = self.tracks[index]
        target = self.targets[index]
        x,y = track.get_selected_audio_seqs(target)
        # toc = time.perf_counter()
        # print("get item",toc-tic)
        # We are taking one channel
        x = x[None]
        y = y[None]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    def __len__(self):
        return len(self.targets)



class SimpleMUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subset='train',
        split='train',
        target='vocals',
        seq_duration=None,
    ):
        """MUSDB18 Dataset wrapper
        """
        self.seq_duration = seq_duration
        self.target = target
        self.mus = musdb.DB(
            download=True,
            split=split,
            subsets=subset,
        )

    def __getitem__(self, index):
        track = self.mus[index]
        track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
        track.chunk_duration = self.seq_duration
        x = track.audio.T
        y = track.targets[self.target].audio.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    def __len__(self):
        return len(self.mus)