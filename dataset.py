import json
import math
import os
import re
import random
import json
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D

def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        print(f"xxx {file_path} xxx")
        return None

# V2C
class Dataset_denoise2_V2Cdataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path_denoise2"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.TopK = train_config["TopK"]

        self.inference_mode = inference_mode
        if self.inference_mode:
            self.basename, self.text, self.speaker, self.basename_ref, self.text_ref = self.inference_process_meta(
                filename
            )
            self.basename_ref = self.basename
            self.text_ref = self.text
        else:
            self.basename, self.speaker, self.text = self.process_meta(
                filename)
            
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = train_config["dataset"]["shuffle_refmel"]
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
        if self.dataset_name == "MovieAnimation":
            print("Reading emotions.json ...")
            with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                self.emotion_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
              
        
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        energy = energy[:min(len(phone),len(duration))]
        pitch = pitch[:min(len(phone),len(duration))]        
        if self.inference_mode:
            basename_ref = self.basename_ref[idx]
            text_ref = self.text_ref[idx]
            ref_mel, ref_linguistic, reordered_array = self.get_random_reference(speaker, basename_ref, text_ref)
        else:
            ref_mel, ref_linguistic, reordered_array = self.get_reference(mel, phone, duration)

        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
        elif self.dataset_name == "MovieAnimation":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_V2C_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
            face_embedding = np.load(face_embedding_path)
            emos_embedding_path = os.path.join(
                self.preprocessed_path,
                "emos",
                "{}-emo-{}.npy".format(speaker, basename),
            )
            emos_embedding = np.load(emos_embedding_path)
            
            emotion_id = self.emotion_map[basename]
            
        elif self.dataset_name == "Grid":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Grid_152_gray",
                "{}-face-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            lip_embedding = np.load(lip_embedding_path)
        
        if mel.shape[0] > sum(duration):
            mel = mel[:sum(duration), :]
        if self.dataset_name == "MovieAnimation":
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                ".npy".format(basename),
            )
        elif self.dataset_name == "Grid":
            spk_path = os.path.join(
                self.preprocessed_path,
                "Grid_spk2",
                "{}-spk-{}.npy".format(speaker, basename),
            )
            
        
            
        spk_embedding = np.load(spk_path)
        
        scene_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "scene_emo_roberta",
            "{}-scene_attri_emo.npy".format(basename),
        )
        scene_emo_embedding = np.load(scene_emo_embedding_path) 
        
        face_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "face_emo_roberta",
            "{}-face_caption.npy".format(basename),
        )
        face_emo_embedding = np.load(face_emo_embedding_path) 
        
        text_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "text_cat_react_emb_reberta",
            "{}-text_cat_react-{}.npy".format(basename, speaker),
        )
        text_emo_embedding= np.load(text_emo_embedding_path)
        
            
        scene_index_file_path = os.path.join(
            "",
            f"{basename}-scene_attri_emo.npy_top_100.json" 
        )
        
        face_index_file_path = os.path.join(
            "",
            f"{basename}-face_caption.npy_top_100.json" 
        )        

        text_index_file_path = os.path.join(
            "",
            f"{basename}_top_100.json" 
        )  

        scene_data = load_json_file(scene_index_file_path)
        face_data = load_json_file(face_index_file_path)
        text_data = load_json_file(text_index_file_path)
        actual_topk = min(self.TopK, len(scene_data), len(face_data), len(text_data))
        
        scene_refs_list = []
        face_refs_list = []
        text_refs_and_list = []
        
        scene_ref_audios_list = []
        face_ref_audios_list = []
        text_ref_audios_list = []
        
        scene_emb_path = ""
        face_emb_path = ""
        text_emb_path = ""
        #text_react_emb_path = ""
        audio_emb_path = ""
        
        for i in range(actual_topk):
            
            scene_file_name_i = scene_data[i]["file_name"]
            
            scene_i_basename = scene_file_name_i.split("-scene_attri_emo")[0]
            scene_i_speaker = scene_i_basename.split("_")[0]
    
            face_file_name_i = face_data[i]["file_name"]
            
            face_i_basename = face_file_name_i.split("-face_caption")[0]
            face_i_speaker = face_i_basename.split("_")[0]
            
            text_file_name_i = text_data[i]["file_name"]
            
            text_i_basename = text_file_name_i.split("-text_emo")[0]
            text_i_speaker = text_i_basename.split("_")[0]
            
                        
            scene_ref_file_path_i = os.path.join(
                scene_emb_path,
                "{}-scene_attri_emo.npy".format(scene_i_basename)
            )
            
            scene_refs_list.append(np.load(scene_ref_file_path_i))
            
            face_ref_file_path_i = os.path.join(
                face_emb_path,
                "{}-face_caption.npy".format(face_i_basename)
            )
            
            face_refs_list.append(np.load(face_ref_file_path_i))
            
            text_ref_file_path_i = os.path.join(
                text_emb_path,
                "{}-text_cat_react-{}.npy".format(text_i_basename, text_i_speaker)
            )

            
            text_refs_and_list.append(np.load(text_ref_file_path_i))
                    
            scene_audio_file_path_i = os.path.join(
                audio_emb_path,
                "{}-audio_emo-{}.npy".format(scene_i_basename, scene_i_speaker)
            )
            scene_ref_audios_list.append(np.load(scene_audio_file_path_i))
            
            face_audio_file_path_i = os.path.join(
                audio_emb_path,
                "{}-audio_emo-{}.npy".format(face_i_basename, face_i_speaker)
            )
            face_ref_audios_list.append(np.load(face_audio_file_path_i))
            
            text_audio_file_path_i = os.path.join(
                audio_emb_path,
                "{}-audio_emo-{}.npy".format(text_i_basename, text_i_speaker)
            )
            text_ref_audios_list.append(np.load(text_audio_file_path_i))            
            
        scene_refs = np.vstack(scene_refs_list)
        face_refs = np.vstack(face_refs_list)
        text_refs = np.vstack(text_refs_and_list)

        scene_ref_audios = np.vstack(scene_ref_audios_list)
        face_ref_audios = np.vstack(face_ref_audios_list)
        text_ref_audios = np.vstack(text_ref_audios_list)      
                       
            
         
                
        
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lip_embedding": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_reordered_array": reordered_array,
            "face_embedding": face_embedding,
            "emos_embedding": emos_embedding,
            "emotion_id": emotion_id,
            "scene_emo_embedding":scene_emo_embedding,
            "face_emo_embedding":face_emo_embedding,
            "text_emo_embedding":text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,               
        }
        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            random.shuffle(index)
            reordered_array = np.take(phone, index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)
            reordered_array = phone

        return mel_post, linguistic, reordered_array

    def get_random_reference(self, speaker, basename_ref, text_ref):
        ref_mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename_ref),
        )
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename_ref),)
        ref_mel = np.load(ref_mel_path)
        duration = np.load(duration_path)
        phone = np.array(text_to_sequence(text_ref, self.cleaners))
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        if ref_mel.shape[0] > sum(duration):
            ref_mel = ref_mel[:sum(duration), :]
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(ref_mel[start: start + duration[i]])
            start += duration[i]

        mel_post = np.concatenate(mel_slices)
        linguistic = np.concatenate(linguistic_slices)
        
        reordered_array = phone

        return mel_post, linguistic, reordered_array


    def inference_process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            basename_aim = []
            phoneme_aim = []
            speaker = []
            basename_ref = []
            phoneme_ref = []
            for line in f.readlines():
                ba, pa, s, bf, pf = line.strip("\n").split("|")
                basename_aim.append(ba)
                phoneme_aim.append(pa)
                speaker.append(s)
                basename_ref.append(bf)
                phoneme_ref.append(pf)
            return basename_aim, phoneme_aim, speaker, basename_ref, phoneme_ref

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, data, idxs):
        emos_embedding = [data[idx]["emos_embedding"] for idx in idxs]
        face_embedding = [data[idx]["face_embedding"] for idx in idxs]
        
        spk_embedding = [data[idx]["spk_embedding"] for idx in idxs]
        scene_emo_embedding = [data[idx]["scene_emo_embedding"] for idx in idxs]
        face_emo_embedding = [data[idx]["face_emo_embedding"] for idx in idxs]
        text_emo_embedding = [data[idx]["text_emo_embedding"] for idx in idxs]
        ids = [data[idx]["id"] for idx in idxs]
        emotion_ids = [data[idx]["emotion_id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        Ref_texts = [data[idx]["Ref_reordered_array"] for idx in idxs]
        raw_texts = texts
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]
        
        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]
        scene_refs = [data[idx]["scene_refs"] for idx in idxs]
        face_refs = [data[idx]["face_refs"] for idx in idxs]
        text_refs = [data[idx]["text_refs"] for idx in idxs]
        scene_ref_audios = [data[idx]["scene_ref_audios"] for idx in idxs]
        face_ref_audios = [data[idx]["face_ref_audios"] for idx in idxs]
        text_ref_audios = [data[idx]["text_ref_audios"] for idx in idxs]
        
        face_lens = np.array([feature_256_e.shape[0] for feature_256_e in lip_embedding])

        text_lens = np.array([text.shape[0] for text in texts])
        
        Ref_text_lens = np.array([Ref_text.shape[0] for Ref_text in Ref_texts])
        
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])

        speakers = np.array(speakers)
        emotion_ids = np.array(emotion_ids)
        texts = pad_1D(texts)
        
        Ref_texts = pad_1D(Ref_texts)
        
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        
        lip_embedding = pad_2D(lip_embedding)

        scene_refs = pad_2D(scene_refs)
        face_refs = pad_2D(face_refs)
        text_refs = pad_2D(text_refs)
        scene_ref_audios = pad_2D(scene_ref_audios)
        face_ref_audios = pad_2D(face_ref_audios)
        text_ref_audios = pad_2D(text_ref_audios)
        
        spk_embedding = np.array(spk_embedding)
        scene_emo_embedding = np.array(scene_emo_embedding) 
        face_emo_embedding = np.array(face_emo_embedding) 
        text_emo_embedding = np.array(text_emo_embedding)  
        
        emos_embedding = np.array(emos_embedding)
        
        face_embedding = pad_2D(face_embedding)
        out = {
            "id": ids,         
            "text": texts,
            "src_len": text_lens,
            "sid": speakers,
            "ref_mels": ref_mels,
            "ref_mel_lens": ref_mel_lens,
            "mel_target": mels,
            "mel_len": mel_lens,
            "f0": pitches,
            "energy": energies,
            "D": durations,
            "ref_linguistics": ref_linguistics,
            "face_lens": face_lens,
            "Lipmotion": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_texts": Ref_texts,
            "Ref_text_lens": Ref_text_lens,
            "emos_embedding": emos_embedding,
            "face_embedding": face_embedding,
            "emotion_ids": emotion_ids,
            "scene_emo_embedding": scene_emo_embedding,
            "face_emo_embedding": face_emo_embedding,
            "text_emo_embedding": text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,    
        }
        
        return out
        

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output



    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path_denoise2"]  # preprocessed_path_denoise2
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.basename, self.text, self.speaker, self.basename_ref, self.text_ref = self.inference_process_meta(
                filename
            )
        else:
            self.basename, self.speaker, self.text = self.process_meta(
                filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = train_config["dataset"]["shuffle_refmel"]
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
        if self.dataset_name == "MovieAnimation":
            print("Reading emotions.json ...")
            with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                self.emotion_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        if self.inference_mode:
            basename_ref = self.basename_ref[idx]
            text_ref = self.text_ref[idx]
            ref_mel, ref_linguistic, reordered_array = self.get_random_reference(speaker, basename_ref, text_ref)

            if self.dataset_name == "MovieAnimation":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "spk2",
                    "xxx-{}.npy".format(basename_ref),
                )
            elif self.dataset_name == "Grid":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "Grid_spk2",
                    "{}-spk-{}.npy".format(speaker, basename_ref),
                )
                
        else:
            ref_mel, ref_linguistic, reordered_array = self.get_reference(mel, phone, duration)

            if self.dataset_name == "MovieAnimation":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "spk2",
                    "xxx-{}.npy".format(basename),
                )
            elif self.dataset_name == "Grid":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "Grid_spk2",
                    "{}-spk-{}.npy".format(speaker, basename),
                )
        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
        elif self.dataset_name == "MovieAnimation":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_V2C_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
            face_embedding = np.load(face_embedding_path)
        elif self.dataset_name == "Grid":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Grid_152_gray",
                "{}-face-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            lip_embedding = np.load(lip_embedding_path)
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "Grid_VA_feature",
                "{}-feature-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            face_embedding = np.load(face_embedding_path)
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        energy = energy[:min(len(phone),len(duration))]
        pitch = pitch[:min(len(phone),len(duration))]
        if mel.shape[0] > sum(duration):
            mel = mel[:sum(duration), :]
        spk_embedding = np.load(spk_path)
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lip_embedding": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_reordered_array": reordered_array,
            "face_embedding": face_embedding,
        }

        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            random.shuffle(index)
            reordered_array = np.take(phone, index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)
            reordered_array = phone
        return mel_post, linguistic, reordered_array

    def get_random_reference(self, speaker, basename_ref, text_ref):
        ref_mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename_ref),
        )
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename_ref),)
        ref_mel = np.load(ref_mel_path)
        duration = np.load(duration_path) 
        phone = np.array(text_to_sequence(text_ref, self.cleaners))
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        if ref_mel.shape[0] > sum(duration):
            ref_mel = ref_mel[:sum(duration), :]
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(ref_mel[start: start + duration[i]])
            start += duration[i]
        mel_post = np.concatenate(mel_slices)
        linguistic = np.concatenate(linguistic_slices)
        reordered_array = phone
        return mel_post, linguistic, reordered_array


    def inference_process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            basename_aim = []
            phoneme_aim = []
            speaker = []
            basename_ref = []
            phoneme_ref = []
            for line in f.readlines():
                ba, pa, s, bf, pf = line.strip("\n").split("|")
                basename_aim.append(ba)
                phoneme_aim.append(pa)
                speaker.append(s)
                basename_ref.append(bf)
                phoneme_ref.append(pf)
            return basename_aim, phoneme_aim, speaker, basename_ref, phoneme_ref

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, data, idxs):
        face_embedding = [data[idx]["face_embedding"] for idx in idxs]
        spk_embedding = [data[idx]["spk_embedding"] for idx in idxs]
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        Ref_texts = [data[idx]["Ref_reordered_array"] for idx in idxs]
        raw_texts = texts
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]
        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]
        face_lens = np.array([feature_256_e.shape[0] for feature_256_e in lip_embedding])
        text_lens = np.array([text.shape[0] for text in texts])
        Ref_text_lens = np.array([Ref_text.shape[0] for Ref_text in Ref_texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        Ref_texts = pad_1D(Ref_texts)
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        lip_embedding = pad_2D(lip_embedding)
        spk_embedding = np.array(spk_embedding)
        face_embedding = pad_2D(face_embedding)
        out = {
            "id": ids,         
            "text": texts,
            "src_len": text_lens,
            "sid": speakers,
            "ref_mels": ref_mels,
            "ref_mel_lens": ref_mel_lens,
            "mel_target": mels,
            "mel_len": mel_lens,
            "f0": pitches,
            "energy": energies,
            "D": durations,
            "ref_linguistics": ref_linguistics,
            "face_lens": face_lens,
            "Lipmotion": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_texts": Ref_texts,
            "Ref_text_lens": Ref_text_lens,
            "face_embedding": face_embedding,
        }
        
        return out
        

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output



    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path_denoise2"]  # preprocessed_path_denoise2
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.basename, self.text, self.speaker, self.basename_ref, self.text_ref = self.inference_process_meta(filename)
        else:
            self.basename, self.speaker, self.text = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = False
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"] 
        if self.dataset_name == "MovieAnimation":
            print("Reading emotions.json ...")
            with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                self.emotion_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        if self.inference_mode:
            basename_ref = self.basename_ref[idx]
            text_ref = self.text_ref[idx]
            ref_mel, ref_linguistic, reordered_array = self.get_random_reference(speaker, basename_ref, text_ref)
            if self.dataset_name == "MovieAnimation":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "spk2",
                    "xxx-spk-{}.npy".format(basename_ref),
                )
            elif self.dataset_name == "Grid":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "Grid_spk2",
                    "{}-spk-{}.npy".format(speaker, basename_ref),
                )
        else:
            ref_mel, ref_linguistic, reordered_array = self.get_reference(mel, phone, duration)
            if self.dataset_name == "MovieAnimation":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "spk2",
                    "xxx_all-spk-{}.npy".format(basename),
                )
            elif self.dataset_name == "Grid":
                spk_path = os.path.join(
                    self.preprocessed_path,
                    "Grid_spk2",
                    "{}-spk-{}.npy".format(speaker, basename),
                )
        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
        elif self.dataset_name == "MovieAnimation":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_V2C_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
            face_embedding = np.load(face_embedding_path)
        elif self.dataset_name == "Grid":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Grid_152_gray",
                "{}-face-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            lip_embedding = np.load(lip_embedding_path)
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "Grid_VA_feature",
                "{}-feature-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            face_embedding = np.load(face_embedding_path)
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        energy = energy[:min(len(phone),len(duration))]
        pitch = pitch[:min(len(phone),len(duration))]
        if mel.shape[0] > sum(duration):
            mel = mel[:sum(duration), :]
        spk_embedding = np.load(spk_path)
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lip_embedding": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_reordered_array": reordered_array,
            "face_embedding": face_embedding,
        }
        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            random.shuffle(index)
            reordered_array = np.take(phone, index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)
            reordered_array = phone

        return mel_post, linguistic, reordered_array

    def get_random_reference(self, speaker, basename_ref, text_ref):
        ref_mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename_ref),
        )
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename_ref),)
        ref_mel = np.load(ref_mel_path)
        duration = np.load(duration_path)
        phone = np.array(text_to_sequence(text_ref, self.cleaners))
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        if ref_mel.shape[0] > sum(duration):
            ref_mel = ref_mel[:sum(duration), :]
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(ref_mel[start: start + duration[i]])
            start += duration[i]
        mel_post = np.concatenate(mel_slices)
        linguistic = np.concatenate(linguistic_slices)
        reordered_array = phone
        return mel_post, linguistic, reordered_array

    def inference_process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            basename_aim = []
            phoneme_aim = []
            speaker = []
            basename_ref = []
            phoneme_ref = []
            for line in f.readlines():
                ba, pa, s, bf, pf = line.strip("\n").split("|")
                basename_aim.append(ba)
                phoneme_aim.append(pa)
                speaker.append(s)
                basename_ref.append(bf)
                phoneme_ref.append(pf)
            return basename_aim, phoneme_aim, speaker, basename_ref, phoneme_ref

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, data, idxs):
        face_embedding = [data[idx]["face_embedding"] for idx in idxs]
        spk_embedding = [data[idx]["spk_embedding"] for idx in idxs]
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        Ref_texts = [data[idx]["Ref_reordered_array"] for idx in idxs]
        raw_texts = texts
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]
        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]
        face_lens = np.array([feature_256_e.shape[0] for feature_256_e in lip_embedding])
        text_lens = np.array([text.shape[0] for text in texts])
        Ref_text_lens = np.array([Ref_text.shape[0] for Ref_text in Ref_texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        Ref_texts = pad_1D(Ref_texts)
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        lip_embedding = pad_2D(lip_embedding)
        spk_embedding = np.array(spk_embedding) 
        face_embedding = pad_2D(face_embedding)
        out = {
            "id": ids,         
            "text": texts,
            "src_len": text_lens,
            "sid": speakers,
            "ref_mels": ref_mels,
            "ref_mel_lens": ref_mel_lens,
            "mel_target": mels,
            "mel_len": mel_lens,
            "f0": pitches,
            "energy": energies,
            "D": durations,
            "ref_linguistics": ref_linguistics,
            "face_lens": face_lens,
            "Lipmotion": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_texts": Ref_texts,
            "Ref_text_lens": Ref_text_lens,
            "face_embedding": face_embedding,
        }
        return out
    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output


    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path_denoise2"]  # preprocessed_path_denoise2
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.inference_mode = inference_mode
        if self.inference_mode:
            self.basename, self.text, self.speaker, self.basename_ref, self.text_ref = self.inference_process_meta(
                filename
            )
        else:
            self.basename, self.speaker, self.text = self.process_meta(
                filename
            )
            
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = train_config["dataset"]["shuffle_refmel"]
        
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        if self.inference_mode:
            basename_ref = self.basename_ref[idx]
            text_ref = self.text_ref[idx]
            
            ref_mel, ref_linguistic, reordered_array = self.get_random_reference(speaker, basename_ref, text_ref)
        else:
            ref_mel, ref_linguistic, reordered_array = self.get_reference(mel, phone, duration)

        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)

        elif self.dataset_name == "MovieAnimation":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_V2C_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
            
            #
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
            face_embedding = np.load(face_embedding_path)
            
        elif self.dataset_name == "Grid":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Grid_152_gray",
                "{}-face-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            lip_embedding = np.load(lip_embedding_path)
            
            #
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "Grid_VA_feature",
                "{}-feature-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            face_embedding = np.load(face_embedding_path)
        
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        energy = energy[:min(len(phone),len(duration))]
        pitch = pitch[:min(len(phone),len(duration))]
        if mel.shape[0] > sum(duration):
            mel = mel[:sum(duration), :]


        # 
        if self.dataset_name == "MovieAnimation":
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "xxx-spk-{}.npy".format(basename_ref),
            )
        elif self.dataset_name == "Grid":
            spk_path = os.path.join(
                self.preprocessed_path,
                "Grid_spk2",
                "{}-spk-{}.npy".format(speaker, basename_ref),
            )

        spk_embedding = np.load(spk_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lip_embedding": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_reordered_array": reordered_array,
            "face_embedding": face_embedding,
        }

        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            random.shuffle(index)
            reordered_array = np.take(phone, index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)

        return mel_post, linguistic, reordered_array

    def get_random_reference(self, speaker, basename_ref, text_ref):
        ref_mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename_ref),
        )
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename_ref),)
        ref_mel = np.load(ref_mel_path)
        duration = np.load(duration_path)
        phone = np.array(text_to_sequence(text_ref, self.cleaners))
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        if ref_mel.shape[0] > sum(duration):
            ref_mel = ref_mel[:sum(duration), :]
        
        
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(ref_mel[start: start + duration[i]])
            start += duration[i]

        mel_post = np.concatenate(mel_slices)
        linguistic = np.concatenate(linguistic_slices)
        
        reordered_array = phone

        return mel_post, linguistic, reordered_array


    def inference_process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            basename_aim = []
            phoneme_aim = []
            speaker = []
            basename_ref = []
            phoneme_ref = []
            for line in f.readlines():
                ba, pa, s, bf, pf = line.strip("\n").split("|")
                basename_aim.append(ba)
                phoneme_aim.append(pa)
                speaker.append(s)
                basename_ref.append(bf)
                phoneme_ref.append(pf)
            return basename_aim, phoneme_aim, speaker, basename_ref, phoneme_ref

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, data, idxs):
        face_embedding = [data[idx]["face_embedding"] for idx in idxs]
        spk_embedding = [data[idx]["spk_embedding"] for idx in idxs]
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        Ref_texts = [data[idx]["Ref_reordered_array"] for idx in idxs]
        raw_texts = texts
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]
        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]
        face_lens = np.array([feature_256_e.shape[0] for feature_256_e in lip_embedding])
        text_lens = np.array([text.shape[0] for text in texts])
        Ref_text_lens = np.array([Ref_text.shape[0] for Ref_text in Ref_texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        Ref_texts = pad_1D(Ref_texts)
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        lip_embedding = pad_2D(lip_embedding)
        spk_embedding = np.array(spk_embedding)
        face_embedding = pad_2D(face_embedding)

        out = {
            "id": ids,         
            "text": texts,
            "src_len": text_lens,
            "sid": speakers,
            "ref_mels": ref_mels,
            "ref_mel_lens": ref_mel_lens,
            "mel_target": mels,
            "mel_len": mel_lens,
            "f0": pitches,
            "energy": energies,
            "D": durations,
            "ref_linguistics": ref_linguistics,
            "face_lens": face_lens,
            "Lipmotion": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_texts": Ref_texts,
            "Ref_text_lens": Ref_text_lens,
            "face_embedding": face_embedding,
        }
        
        return out
        

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path_denoise2"]  # preprocessed_path_denoise2
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.basename, self.text, self.speaker, self.basename_ref, self.text_ref = self.inference_process_meta(
                filename)
        else:
            self.basename, self.speaker, self.text = self.process_meta(
                filename) 
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = train_config["dataset"]["shuffle_refmel"]
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
        if self.dataset_name == "MovieAnimation":
            print("Reading emotions.json ...")
            with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                self.emotion_map = json.load(f)
        self.TopK = train_config["TopK"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),)
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        if self.inference_mode:
            basename_ref = self.basename_ref[idx]
            text_ref = self.text_ref[idx]
            ref_mel, ref_linguistic, reordered_array = self.get_random_reference(speaker, basename_ref, text_ref)
        else:
            ref_mel, ref_linguistic, reordered_array = self.get_reference(mel, phone, duration)
        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
        elif self.dataset_name == "MovieAnimation":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_V2C_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
            face_embedding = np.load(face_embedding_path)
            emos_embedding_path = os.path.join(
                self.preprocessed_path,
                "emos",
                "{}-emo-{}.npy".format(speaker, basename),
            )
            emos_embedding = np.load(emos_embedding_path)
            emotion_id = self.emotion_map[basename]
            
        elif self.dataset_name == "Grid":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Grid_152_gray",
                "{}-face-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            lip_embedding = np.load(lip_embedding_path)
        
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        energy = energy[:min(len(phone),len(duration))]
        pitch = pitch[:min(len(phone),len(duration))]
        if mel.shape[0] > sum(duration):
            mel = mel[:sum(duration), :]
        if self.dataset_name == "MovieAnimation":
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "xxx-spk-{}.npy".format(basename_ref),
            )
        elif self.dataset_name == "Grid":
            spk_path = os.path.join(
                self.preprocessed_path,
                "Grid_spk2",
                "{}-spk-{}.npy".format(speaker, basename_ref),
            )
        spk_embedding = np.load(spk_path)

        scene_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "scene_caption_embedding",
            "{}-scene_caption-{}.npy".format(basename, speaker),
        )
        scene_emo_embedding = np.load(scene_emo_embedding_path) 
        
        face_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "face_caption_embedding",
            "{}-face_caption-{}.npy".format(basename, speaker),
        )
        face_emo_embedding = np.load(face_emo_embedding_path) 
        
        text_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "text_emo_embedding",
            "{}-text_emo-{}.npy".format(basename, speaker),
        )
        text_emo_embedding = np.load(text_emo_embedding_path)
            
        scene_index_file_path = os.path.join(
            "xxx",
            f"{basename}-scene_caption-{speaker}.npy_top_6500.json" 
        )
        
        face_index_file_path = os.path.join(
            "xxx",
            f"{basename}-face_caption-{speaker}.npy_top_6500.json" 
        )        

        text_index_file_path = os.path.join(
            "xxx",
            f"{basename}-text_emo-{speaker}.npy_top_6500.json" 
        )  

        # 导入这三个 JSON 文件
        scene_data = load_json_file(scene_index_file_path)
        face_data = load_json_file(face_index_file_path)
        text_data = load_json_file(text_index_file_path)
        
        scene_refs_list = []
        face_refs_list = []
        text_refs_list = []
        
        scene_ref_audios_list = []
        face_ref_audios_list = []
        text_ref_audios_list = []
        
        scene_emb_path = "xxx"
        face_emb_path = "xxx"
        text_emb_path = "xxx"
        audio_emb_path = "xxx"
        
        for i in range(self.TopK):
            
            scene_file_name_i = scene_data[i]["file_name"]
            face_file_name_i = face_data[i]["file_name"]
            text_file_name_i = text_data[i]["file_name"]
            
            scene_file_name_i_audio = scene_file_name_i.replace('scene_caption', 'audio_caption')
            face_file_name_i_audio = face_file_name_i.replace('face_caption', 'audio_caption')
            text_file_name_i_audio = text_file_name_i.replace('text_emo', 'audio_caption')
                        
            scene_ref_file_path_i = os.path.join(
                scene_emb_path,
                scene_file_name_i
            )
            
            scene_refs_list.append(np.load(scene_ref_file_path_i))
            
            face_ref_file_path_i = os.path.join(
                face_emb_path,
                face_file_name_i
            )
            face_refs_list.append(np.load(face_ref_file_path_i))
            text_ref_file_path_i = os.path.join(
                text_emb_path,
                text_file_name_i
            )
            text_refs_list.append(np.load(text_ref_file_path_i))
                    
            scene_audio_file_path_i = os.path.join(
                audio_emb_path,
                scene_file_name_i_audio
            )
            scene_ref_audios_list.append(np.load(scene_audio_file_path_i))
            
            face_audio_file_path_i = os.path.join(
                audio_emb_path,
                face_file_name_i_audio
            )
            face_ref_audios_list.append(np.load(face_audio_file_path_i))
            
            text_audio_file_path_i = os.path.join(
                audio_emb_path,
                text_file_name_i_audio
            )
            text_ref_audios_list.append(np.load(text_audio_file_path_i))            
            
        scene_refs = np.vstack(scene_refs_list)
        face_refs = np.vstack(face_refs_list)
        text_refs = np.vstack(text_refs_list)

        scene_ref_audios = np.vstack(scene_ref_audios_list)
        face_ref_audios = np.vstack(face_ref_audios_list)
        text_ref_audios = np.vstack(text_ref_audios_list)   

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lip_embedding": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_reordered_array": reordered_array,
            "face_embedding": face_embedding,
            "emos_embedding": emos_embedding,
            "emotion_id": emotion_id,
            "scene_emo_embedding":scene_emo_embedding,
            "face_emo_embedding":face_emo_embedding,
            "text_emo_embedding":text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,             
        }

        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            random.shuffle(index)
            reordered_array = np.take(phone, index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)
        return mel_post, linguistic, reordered_array

    def get_random_reference(self, speaker, basename_ref, text_ref):
        ref_mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename_ref),
        )
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename_ref),)
        ref_mel = np.load(ref_mel_path)
        duration = np.load(duration_path)
        phone = np.array(text_to_sequence(text_ref, self.cleaners))
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        if ref_mel.shape[0] > sum(duration):
            ref_mel = ref_mel[:sum(duration), :]
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(ref_mel[start: start + duration[i]])
            start += duration[i]
        mel_post = np.concatenate(mel_slices)
        linguistic = np.concatenate(linguistic_slices)
        reordered_array = phone
        return mel_post, linguistic, reordered_array

    def inference_process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            basename_aim = []
            phoneme_aim = []
            speaker = []
            basename_ref = []
            phoneme_ref = []
            for line in f.readlines():
                ba, pa, s, bf, pf = line.strip("\n").split("|")
                
                basename_aim.append(ba)
                phoneme_aim.append(pa)
                speaker.append(s)
                basename_ref.append(bf)
                phoneme_ref.append(pf)
            return basename_aim, phoneme_aim, speaker, basename_ref, phoneme_ref

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, data, idxs):
        emos_embedding = [data[idx]["emos_embedding"] for idx in idxs]
        face_embedding = [data[idx]["face_embedding"] for idx in idxs]
        spk_embedding = [data[idx]["spk_embedding"] for idx in idxs]
        scene_emo_embedding = [data[idx]["scene_emo_embedding"] for idx in idxs]
        face_emo_embedding = [data[idx]["face_emo_embedding"] for idx in idxs]
        text_emo_embedding = [data[idx]["text_emo_embedding"] for idx in idxs]
        ids = [data[idx]["id"] for idx in idxs]
        emotion_ids = [data[idx]["emotion_id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        Ref_texts = [data[idx]["Ref_reordered_array"] for idx in idxs]
        raw_texts = texts
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]
        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]

        scene_refs = [data[idx]["scene_refs"] for idx in idxs]
        face_refs = [data[idx]["face_refs"] for idx in idxs]
        text_refs = [data[idx]["text_refs"] for idx in idxs]
        scene_ref_audios = [data[idx]["scene_ref_audios"] for idx in idxs]
        face_ref_audios = [data[idx]["face_ref_audios"] for idx in idxs]
        text_ref_audios = [data[idx]["text_ref_audios"] for idx in idxs]
        face_lens = np.array([feature_256_e.shape[0] for feature_256_e in lip_embedding])
        text_lens = np.array([text.shape[0] for text in texts])
        Ref_text_lens = np.array([Ref_text.shape[0] for Ref_text in Ref_texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])
        speakers = np.array(speakers)
        emotion_ids = np.array(emotion_ids)
        texts = pad_1D(texts)
        Ref_texts = pad_1D(Ref_texts)
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        lip_embedding = pad_2D(lip_embedding)
        spk_embedding = np.array(spk_embedding)
        emos_embedding = np.array(emos_embedding)
        face_embedding = pad_2D(face_embedding)

        scene_emo_embedding = np.array(scene_emo_embedding) 
        face_emo_embedding = np.array(face_emo_embedding) 
        text_emo_embedding = np.array(text_emo_embedding)  
        scene_refs = pad_2D(scene_refs)
        face_refs = pad_2D(face_refs)
        text_refs = pad_2D(text_refs)
        scene_ref_audios = pad_2D(scene_ref_audios)
        face_ref_audios = pad_2D(face_ref_audios)
        text_ref_audios = pad_2D(text_ref_audios)

        out = {
            "id": ids,         
            "text": texts,
            "src_len": text_lens,
            "sid": speakers,
            "ref_mels": ref_mels,
            "ref_mel_lens": ref_mel_lens,
            "mel_target": mels,
            "mel_len": mel_lens,
            "f0": pitches,
            "energy": energies,
            "D": durations,
            "ref_linguistics": ref_linguistics,
            "face_lens": face_lens,
            "Lipmotion": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_texts": Ref_texts,
            "Ref_text_lens": Ref_text_lens,
            "emos_embedding": emos_embedding,
            "face_embedding": face_embedding,
            "emotion_ids": emotion_ids,
            "scene_emo_embedding": scene_emo_embedding,
            "face_emo_embedding": face_emo_embedding,
            "text_emo_embedding": text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,    
        }
        return out
        

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

# need eva1
class Dataset_denoise2_Setting1_Run(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path_denoise2"]  # preprocessed_path_denoise2
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.basename, self.text, self.speaker, self.basename_ref, self.text_ref = self.inference_process_meta(
                filename
            )
        else:
            self.basename, self.speaker, self.text = self.process_meta(
                filename
            )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = False
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
        if self.dataset_name == "MovieAnimation":
            print("Reading emotions.json ...")
            with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                self.emotion_map = json.load(f)
        
        self.TopK = train_config["TopK"]
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        if self.inference_mode:
            basename_ref = self.basename_ref[idx]
            text_ref = self.text_ref[idx]
            ref_mel, ref_linguistic, reordered_array = self.get_random_reference(speaker, basename_ref, text_ref)
        else:
            ref_mel, ref_linguistic, reordered_array = self.get_reference(mel, phone, duration)
        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
        elif self.dataset_name == "MovieAnimation":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_V2C_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
            face_embedding = np.load(face_embedding_path)
            emos_embedding_path = os.path.join(
                self.preprocessed_path,
                "emos",
                "{}-emo-{}.npy".format(speaker, basename),
            )
            emos_embedding = np.load(emos_embedding_path)
            emotion_id = self.emotion_map[basename]
            
        elif self.dataset_name == "Grid":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Grid_152_gray",
                "{}-face-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            lip_embedding = np.load(lip_embedding_path)
        
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        energy = energy[:min(len(phone),len(duration))]
        pitch = pitch[:min(len(phone),len(duration))]
        if mel.shape[0] > sum(duration):
            mel = mel[:sum(duration), :]
        if self.dataset_name == "MovieAnimation":
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "xxx-{}.npy".format(basename),)
        spk_embedding = np.load(spk_path)
        
        scene_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "scene_emo_roberta",
            "{}-scene_attri_emo.npy".format(basename),
        )
        scene_emo_embedding = np.load(scene_emo_embedding_path) 
        
        face_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "face_emo_roberta",
            "{}-face_caption.npy".format(basename),
        )
        face_emo_embedding = np.load(face_emo_embedding_path) 
        
        text_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "text_cat_react_emb_reberta",
            "{}-text_cat_react-{}.npy".format(basename, speaker),
        )
        text_emo_embedding= np.load(text_emo_embedding_path)
        
            
        scene_index_file_path = os.path.join(
            "xxx",
            f"{basename}-scene_attri_emo.npy_top_100.json" 
        )
        
        face_index_file_path = os.path.join(
            "xxx",
            f"{basename}-face_caption.npy_top_100.json" 
        )        

        text_index_file_path = os.path.join(
            "xxx",
            f"{basename}_top_100.json" 
        )    

        scene_data = load_json_file(scene_index_file_path)
        face_data = load_json_file(face_index_file_path)
        text_data = load_json_file(text_index_file_path)
        
        actual_topk = min(self.TopK, len(scene_data), len(face_data), len(text_data))
        
        scene_refs_list = []
        face_refs_list = []
        text_refs_and_list = []
        
        scene_ref_audios_list = []
        face_ref_audios_list = []
        text_ref_audios_list = []
        
        scene_emb_path = "xxx"
        face_emb_path = "xxx"
        text_emb_path = "xxx"
        audio_emb_path = "xxx"
        
        for i in range(actual_topk):
            
            scene_file_name_i = scene_data[i]["file_name"]
            
            scene_i_basename = scene_file_name_i.split("-scene_attri_emo")[0]
            scene_i_speaker = scene_i_basename.split("_")[0]
    
            face_file_name_i = face_data[i]["file_name"]
            
            face_i_basename = face_file_name_i.split("-face_caption")[0]
            face_i_speaker = face_i_basename.split("_")[0]
            
            text_file_name_i = text_data[i]["file_name"]
            
            text_i_basename = text_file_name_i.split("-text_emo")[0]
            text_i_speaker = text_i_basename.split("_")[0]
            
                        
            scene_ref_file_path_i = os.path.join(
                scene_emb_path,
                "{}-scene_attri_emo.npy".format(scene_i_basename)
            )
            
            scene_refs_list.append(np.load(scene_ref_file_path_i))
            
            face_ref_file_path_i = os.path.join(
                face_emb_path,
                "{}-face_caption.npy".format(face_i_basename)
            )
            
            face_refs_list.append(np.load(face_ref_file_path_i))
            
            text_ref_file_path_i = os.path.join(
                text_emb_path,
                "{}-text_cat_react-{}.npy".format(text_i_basename, text_i_speaker)
            )

            
            text_refs_and_list.append(np.load(text_ref_file_path_i))
                    
            scene_audio_file_path_i = os.path.join(
                audio_emb_path,
                "{}-audio_emo-{}.npy".format(scene_i_basename, scene_i_speaker)
            )
            scene_ref_audios_list.append(np.load(scene_audio_file_path_i))
            
            face_audio_file_path_i = os.path.join(
                audio_emb_path,
                "{}-audio_emo-{}.npy".format(face_i_basename, face_i_speaker)
            )
            face_ref_audios_list.append(np.load(face_audio_file_path_i))
            
            text_audio_file_path_i = os.path.join(
                audio_emb_path,
                "{}-audio_emo-{}.npy".format(text_i_basename, text_i_speaker)
            )
            text_ref_audios_list.append(np.load(text_audio_file_path_i))            
            
        scene_refs = np.vstack(scene_refs_list)
        face_refs = np.vstack(face_refs_list)
        text_refs = np.vstack(text_refs_and_list)

        scene_ref_audios = np.vstack(scene_ref_audios_list)
        face_ref_audios = np.vstack(face_ref_audios_list)
        text_ref_audios = np.vstack(text_ref_audios_list)       
        
        
        
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lip_embedding": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_reordered_array": reordered_array,
            "face_embedding": face_embedding,
            "emos_embedding": emos_embedding,
            "emotion_id": emotion_id,
            "scene_emo_embedding": scene_emo_embedding,
            "face_emo_embedding": face_emo_embedding,
            "text_emo_embedding": text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,   
        }

        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            random.shuffle(index)
            reordered_array = np.take(phone, index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)
            reordered_array = phone
        return mel_post, linguistic, reordered_array
    def get_random_reference(self, speaker, basename_ref, text_ref):
        ref_mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename_ref),
        )
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename_ref),)
        ref_mel = np.load(ref_mel_path)
        duration = np.load(duration_path)
        phone = np.array(text_to_sequence(text_ref, self.cleaners))
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        if ref_mel.shape[0] > sum(duration):
            ref_mel = ref_mel[:sum(duration), :]
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(ref_mel[start: start + duration[i]])
            start += duration[i]
        mel_post = np.concatenate(mel_slices)
        linguistic = np.concatenate(linguistic_slices)
        reordered_array = phone
        return mel_post, linguistic, reordered_array


    def inference_process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            basename_aim = []
            phoneme_aim = []
            speaker = []
            basename_ref = []
            phoneme_ref = []
            for line in f.readlines():
                ba, pa, s, bf, pf = line.strip("\n").split("|")
                basename_aim.append(ba)
                phoneme_aim.append(pa)
                speaker.append(s)
                basename_ref.append(bf)
                phoneme_ref.append(pf)
            return basename_aim, phoneme_aim, speaker, basename_ref, phoneme_ref

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, data, idxs):
        emos_embedding = [data[idx]["emos_embedding"] for idx in idxs]
        face_embedding = [data[idx]["face_embedding"] for idx in idxs]
        
        spk_embedding = [data[idx]["spk_embedding"] for idx in idxs]
        ids = [data[idx]["id"] for idx in idxs]
        
        emotion_ids = [data[idx]["emotion_id"] for idx in idxs]
        
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        
        Ref_texts = [data[idx]["Ref_reordered_array"] for idx in idxs]
        
        raw_texts = texts
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]
        
        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]
        
        face_lens = np.array([feature_256_e.shape[0] for feature_256_e in lip_embedding])

        text_lens = np.array([text.shape[0] for text in texts])
        
        Ref_text_lens = np.array([Ref_text.shape[0] for Ref_text in Ref_texts])
        
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])

        speakers = np.array(speakers)
        emotion_ids = np.array(emotion_ids)
        texts = pad_1D(texts)
        
        Ref_texts = pad_1D(Ref_texts)
        
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        
        lip_embedding = pad_2D(lip_embedding)
        
        spk_embedding = np.array(spk_embedding) 
        
        emos_embedding = np.array(emos_embedding) 
        
        face_embedding = pad_2D(face_embedding)
        
        scene_emo_embedding = [data[idx]["scene_emo_embedding"] for idx in idxs]
        face_emo_embedding = [data[idx]["face_emo_embedding"] for idx in idxs]
        text_emo_embedding = [data[idx]["text_emo_embedding"] for idx in idxs]
        scene_refs = [data[idx]["scene_refs"] for idx in idxs]
        face_refs = [data[idx]["face_refs"] for idx in idxs]
        text_refs = [data[idx]["text_refs"] for idx in idxs]
        scene_ref_audios = [data[idx]["scene_ref_audios"] for idx in idxs]
        face_ref_audios = [data[idx]["face_ref_audios"] for idx in idxs]
        text_ref_audios = [data[idx]["text_ref_audios"] for idx in idxs]

        scene_emo_embedding = np.array(scene_emo_embedding) 
        face_emo_embedding = np.array(face_emo_embedding) 
        text_emo_embedding = np.array(text_emo_embedding)  
        scene_refs = pad_2D(scene_refs)
        face_refs = pad_2D(face_refs)
        text_refs = pad_2D(text_refs)
        scene_ref_audios = pad_2D(scene_ref_audios)
        face_ref_audios = pad_2D(face_ref_audios)
        text_ref_audios = pad_2D(text_ref_audios)


        out = {
            "id": ids,         
            "text": texts,
            "src_len": text_lens,
            "sid": speakers,
            "ref_mels": ref_mels,
            "ref_mel_lens": ref_mel_lens,
            "mel_target": mels,
            "mel_len": mel_lens,
            "f0": pitches,
            "energy": energies,
            "D": durations,
            "ref_linguistics": ref_linguistics,
            "face_lens": face_lens,
            "Lipmotion": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_texts": Ref_texts,
            "Ref_text_lens": Ref_text_lens,
            "emos_embedding": emos_embedding,
            "face_embedding": face_embedding,
            "emotion_ids": emotion_ids,
            "scene_emo_embedding": scene_emo_embedding,
            "face_emo_embedding": face_emo_embedding,
            "text_emo_embedding": text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,  
        }
        
        return out
        

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output



    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path_denoise2"]  # preprocessed_path_denoise2
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.inference_mode = inference_mode
        if self.inference_mode:
            self.basename, self.text, self.basename_ref, self.text_ref = self.inference_process_meta(
                filename
            )
        else:
            self.basename, self.speaker, self.text = self.process_meta(
                filename
            )
            
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = train_config["dataset"]["shuffle_refmel"]
        
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
        
        if self.dataset_name == "MovieAnimation":
            print("Reading emotions.json ...")
            with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                self.emotion_map = json.load(f)
        
        self.TopK = train_config["TopK"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = basename.split('_00')[0]
        speaker_id = self.speaker_map[speaker]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        
        basename_ref = self.basename_ref[idx]
        text_ref = self.text_ref[idx]

        if self.inference_mode:
            basename_ref = self.basename_ref[idx]
            text_ref = self.text_ref[idx]
            
            ref_mel, ref_linguistic, reordered_array = self.get_random_reference(speaker, basename_ref, text_ref)
        else:
            ref_mel, ref_linguistic, reordered_array = self.get_reference(mel, phone, duration)

        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
        # extrated_embedding_V2C_gray
        elif self.dataset_name == "MovieAnimation":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_V2C_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)
            
            #
            face_embedding_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
            face_embedding = np.load(face_embedding_path)
            
            #
            emos_embedding_path = os.path.join(
                self.preprocessed_path,
                "emos",
                "{}-emo-{}.npy".format(speaker, basename),
            )
            emos_embedding = np.load(emos_embedding_path)
            
            emotion_id = self.emotion_map[basename]
            
        elif self.dataset_name == "Grid":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Grid_152_gray",
                "{}-face-{}.npy".format(speaker, basename.split(speaker+'-')[-1]),
            )
            lip_embedding = np.load(lip_embedding_path)
        
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        energy = energy[:min(len(phone),len(duration))]
        pitch = pitch[:min(len(phone),len(duration))]
        if mel.shape[0] > sum(duration):
            mel = mel[:sum(duration), :]

        spk_path = os.path.join(
             "/xxx",
            "Grid_spk2",
            "{}-spk-{}.npy".format(basename_ref.split('-')[0], basename_ref),
        )
        spk_embedding = np.load(spk_path)

        scene_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "scene_caption_embedding",
            "{}-scene_caption-{}.npy".format(basename, speaker),
        )
        scene_emo_embedding = np.load(scene_emo_embedding_path) 
        
        face_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "face_caption_embedding",
            "{}-face_caption-{}.npy".format(basename, speaker),
        )
        face_emo_embedding = np.load(face_emo_embedding_path) 
        
        text_emo_embedding_path = os.path.join(
            self.preprocessed_path,
            "text_emo_embedding",
            "{}-text_emo-{}.npy".format(basename, speaker),
        )
        text_emo_embedding = np.load(text_emo_embedding_path)
            
        scene_index_file_path = os.path.join(
            "xxx",
            f"{basename}-scene_caption-{speaker}.npy_top_6500.json" 
        )
        
        face_index_file_path = os.path.join(
            "xxx",
            f"{basename}-face_caption-{speaker}.npy_top_6500.json" 
        )        

        text_index_file_path = os.path.join(
            "/xxx",
            f"{basename}-text_emo-{speaker}.npy_top_6500.json" 
        )  

        scene_data = load_json_file(scene_index_file_path)
        face_data = load_json_file(face_index_file_path)
        text_data = load_json_file(text_index_file_path)
        
        scene_refs_list = []
        face_refs_list = []
        text_refs_list = []
        
        scene_ref_audios_list = []
        face_ref_audios_list = []
        text_ref_audios_list = []
        
        scene_emb_path = "xxx"
        face_emb_path = "xxx"
        text_emb_path = "xxx"
        audio_emb_path = "xxx"
        
        for i in range(self.TopK):
            
            scene_file_name_i = scene_data[i]["file_name"]
            face_file_name_i = face_data[i]["file_name"]
            text_file_name_i = text_data[i]["file_name"]
            
            scene_file_name_i_audio = scene_file_name_i.replace('scene_caption', 'audio_caption')
            face_file_name_i_audio = face_file_name_i.replace('face_caption', 'audio_caption')
            text_file_name_i_audio = text_file_name_i.replace('text_emo', 'audio_caption')
                        
            scene_ref_file_path_i = os.path.join(
                scene_emb_path,
                scene_file_name_i
            )
            
            scene_refs_list.append(np.load(scene_ref_file_path_i))
            
            face_ref_file_path_i = os.path.join(
                face_emb_path,
                face_file_name_i
            )
            face_refs_list.append(np.load(face_ref_file_path_i))
            text_ref_file_path_i = os.path.join(
                text_emb_path,
                text_file_name_i
            )
            text_refs_list.append(np.load(text_ref_file_path_i))
                    
            scene_audio_file_path_i = os.path.join(
                audio_emb_path,
                scene_file_name_i_audio
            )
            scene_ref_audios_list.append(np.load(scene_audio_file_path_i))
            
            face_audio_file_path_i = os.path.join(
                audio_emb_path,
                face_file_name_i_audio
            )
            face_ref_audios_list.append(np.load(face_audio_file_path_i))
            
            text_audio_file_path_i = os.path.join(
                audio_emb_path,
                text_file_name_i_audio
            )
            text_ref_audios_list.append(np.load(text_audio_file_path_i))            
            
        scene_refs = np.vstack(scene_refs_list)
        face_refs = np.vstack(face_refs_list)
        text_refs = np.vstack(text_refs_list)

        scene_ref_audios = np.vstack(scene_ref_audios_list)
        face_ref_audios = np.vstack(face_ref_audios_list)
        text_ref_audios = np.vstack(text_ref_audios_list)

        sample = {
            "id": basename,
            "zeroref": basename_ref,
            "speaker": speaker_id,
            "text": phone,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lip_embedding": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_reordered_array": reordered_array,
            "face_embedding": face_embedding,
            "emos_embedding": emos_embedding,
            "emotion_id": emotion_id,
            "scene_emo_embedding": scene_emo_embedding,
            "face_emo_embedding": face_emo_embedding,
            "text_emo_embedding": text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,   
        }

        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]
        # random shuffle by phoneme duration
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            random.shuffle(index)
            reordered_array = np.take(phone, index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)

        return mel_post, linguistic, reordered_array

    def get_random_reference(self, speaker, basename_ref, text_ref):
        Grid_path = ""xxx""
        ref_mel_path = os.path.join(
            Grid_path,
            "mel",
            "{}-mel-{}.npy".format(basename_ref.split('-')[0], basename_ref),
        )
        duration_path = os.path.join(
            Grid_path,
            "duration",
            "{}-duration-{}.npy".format(basename_ref.split('-')[0], basename_ref),)
        # 
        ref_mel = np.load(ref_mel_path)
        duration = np.load(duration_path)
        
        phone = np.array(text_to_sequence(text_ref, self.cleaners))
        phone = phone[:min(len(phone),len(duration))]
        duration = duration[:min(len(phone),len(duration))]
        if ref_mel.shape[0] > sum(duration):
            ref_mel = ref_mel[:sum(duration), :]
        
        
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(ref_mel[start: start + duration[i]])
            start += duration[i]

        mel_post = np.concatenate(mel_slices)
        linguistic = np.concatenate(linguistic_slices)
        
        reordered_array = phone

        return mel_post, linguistic, reordered_array


    def inference_process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            basename_aim = []
            phoneme_aim = []
            basename_ref = []
            phoneme_ref = []
            for line in f.readlines():
                ba, pa, bf, pf = line.strip("\n").split("|")
                basename_aim.append(ba)
                phoneme_aim.append(pa)
                basename_ref.append(bf)
                phoneme_ref.append(pf)
            return basename_aim, phoneme_aim, basename_ref, phoneme_ref

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, data, idxs):
        emos_embedding = [data[idx]["emos_embedding"] for idx in idxs]
        face_embedding = [data[idx]["face_embedding"] for idx in idxs]
        
        spk_embedding = [data[idx]["spk_embedding"] for idx in idxs]
        scene_emo_embedding = [data[idx]["scene_emo_embedding"] for idx in idxs]
        face_emo_embedding = [data[idx]["face_emo_embedding"] for idx in idxs]
        text_emo_embedding = [data[idx]["text_emo_embedding"] for idx in idxs]
        ids = [data[idx]["id"] for idx in idxs]
        zerorefs = [data[idx]["zeroref"] for idx in idxs]
        emotion_ids = [data[idx]["emotion_id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        Ref_texts = [data[idx]["Ref_reordered_array"] for idx in idxs]
        raw_texts = texts
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]
        
        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]

        scene_refs = [data[idx]["scene_refs"] for idx in idxs]
        face_refs = [data[idx]["face_refs"] for idx in idxs]
        text_refs = [data[idx]["text_refs"] for idx in idxs]
        scene_ref_audios = [data[idx]["scene_ref_audios"] for idx in idxs]
        face_ref_audios = [data[idx]["face_ref_audios"] for idx in idxs]
        text_ref_audios = [data[idx]["text_ref_audios"] for idx in idxs]
        
        face_lens = np.array([feature_256_e.shape[0] for feature_256_e in lip_embedding])

        text_lens = np.array([text.shape[0] for text in texts])
        
        Ref_text_lens = np.array([Ref_text.shape[0] for Ref_text in Ref_texts])
        
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])

        speakers = np.array(speakers)
        emotion_ids = np.array(emotion_ids)
        texts = pad_1D(texts)
        
        Ref_texts = pad_1D(Ref_texts)
        
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        lip_embedding = pad_2D(lip_embedding)
        spk_embedding = np.array(spk_embedding) 
        emos_embedding = np.array(emos_embedding) 
        face_embedding = pad_2D(face_embedding)

        scene_emo_embedding = np.array(scene_emo_embedding) 
        face_emo_embedding = np.array(face_emo_embedding) 
        text_emo_embedding = np.array(text_emo_embedding)  
        scene_refs = pad_2D(scene_refs)
        face_refs = pad_2D(face_refs)
        text_refs = pad_2D(text_refs)
        scene_ref_audios = pad_2D(scene_ref_audios)
        face_ref_audios = pad_2D(face_ref_audios)
        text_ref_audios = pad_2D(text_ref_audios)

        out = {
            "id": ids,  
            "zerorefs": zerorefs,       
            "text": texts,
            "src_len": text_lens,
            "sid": speakers,
            "ref_mels": ref_mels,
            "ref_mel_lens": ref_mel_lens,
            "mel_target": mels,
            "mel_len": mel_lens,
            "f0": pitches,
            "energy": energies,
            "D": durations,
            "ref_linguistics": ref_linguistics,
            "face_lens": face_lens,
            "Lipmotion": lip_embedding,
            "spk_embedding": spk_embedding,
            "Ref_texts": Ref_texts,
            "Ref_text_lens": Ref_text_lens,
            "emos_embedding": emos_embedding,
            "face_embedding": face_embedding,
            "emotion_ids": emotion_ids,
            "scene_emo_embedding": scene_emo_embedding,
            "face_emo_embedding": face_emo_embedding,
            "text_emo_embedding": text_emo_embedding,
            "scene_refs": scene_refs,
            "face_refs": face_refs,
            "text_refs": text_refs,
            "scene_ref_audios": scene_ref_audios,
            "face_ref_audios": face_ref_audios,    
            "text_ref_audios": text_ref_audios,   
        }
        
        return out
        

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
