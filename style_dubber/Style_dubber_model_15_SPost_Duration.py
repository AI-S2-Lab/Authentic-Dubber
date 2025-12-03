"""
Note: 
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Decoder_Condition, PostNet, Lip_Encoder
from .modules import VarianceAdaptor_softplus1
from utils.tools import get_mask_from_lengths
from MMAttention.sma import StepwiseMonotonicMultiheadAttention

from style_dubber.modules import AdversarialClassifier
import yaml
import math

from .Graph_utile import RGAT

class Basic_Emotion_Graph(nn.Module):
    "Basic_Emotion_Graph"
    
    def __init__(self, hyp_params, ):
        super(Basic_Emotion_Graph, self).__init__()
        
        self.dim = hyp_params['orig_d_l']


        self.RGAT = RGAT(nfeat=self.dim, nhid=self.dim, num_relation=2)
        self.linear_scene = nn.Linear(1024, 256)
        self.linear_face = nn.Linear(1024, 256)
        self.linear_text = nn.Linear(2048, 256)
        
    def build_adj_total(self, graph_nodes):

        batch_size, total_nodes = graph_nodes.size(0), graph_nodes.size(1)

        adj_matrices = torch.zeros(
            (batch_size, total_nodes, total_nodes), 
            dtype=torch.long, 
            device=graph_nodes.device
        )


        adj_matrices[:, 0, 1] = adj_matrices[:, 1, 0] = 1  
        adj_matrices[:, 0, 2] = adj_matrices[:, 2, 0] = 1  
        adj_matrices[:, 1, 2] = adj_matrices[:, 2, 1] = 1  

        return adj_matrices

    def forward(self, 
                scene_treivel_source = None,
                face_treivel_source = None,
                text_treivel_source = None,
                ):
        
        
        scene = self.linear_scene(scene_treivel_source)
        face = self.linear_face(face_treivel_source)
        text = self.linear_text(text_treivel_source)
        graph_nodes = torch.cat([scene, face, text], dim=1) 
                       
        adj_total = self.build_adj_total(graph_nodes)

        Graph_Stage_1_output = self.RGAT(graph_nodes, adj_total)         
        return Graph_Stage_1_output
    


    



class Indirect_Emotion_Extended_Graph(nn.Module):
    "Indirect_Emotion_Extended_Graph"
    
    def __init__(self, hyp_params):
        super(Indirect_Emotion_Extended_Graph, self).__init__()
        
        self.dim = hyp_params['orig_d_l']

        self.RGAT = RGAT(nfeat=self.dim, nhid=self.dim, num_relation=3)
        self.linear_scene = nn.Linear(1024, 256)
        self.linear_face = nn.Linear(1024, 256)
        self.linear_text = nn.Linear(2048, 256)

    def forward(self, 
                graph_stage_1_output=None,
                scene_refs=None, 
                face_refs=None, 
                text_refs=None):
                

        scene_refs_256 = self.linear_scene(scene_refs)
        face_refs_256 = self.linear_face(face_refs)
        text_refs_256 = self.linear_text(text_refs)
        

        graph_nodes = torch.cat([graph_stage_1_output, scene_refs_256, face_refs_256, text_refs_256], dim=1)


        adj_matrix = self.build_adjacency_matrix(scene_refs_256)
        
        Graph_Stage_2_output = self.RGAT(graph_nodes, adj_matrix)       
        
        return Graph_Stage_2_output
    
    def build_adjacency_matrix(self,scene_refs_256):
        """

        """

        K = scene_refs_256.size(1)  
        batch_size = scene_refs_256.size(0)
   
        total_nodes = 3 + 3 * K
        adj_matrix = torch.zeros((batch_size, total_nodes, total_nodes), dtype=torch.long, device=scene_refs_256.device)
   
        adj_matrix[:,0, 1] = adj_matrix[:,1, 0] = 1  
        adj_matrix[:,0, 2] = adj_matrix[:,2, 0] = 1  
        adj_matrix[:,1, 2] = adj_matrix[:,2, 1] = 1  


        for i in range(K):
            adj_matrix[:,0, 3 + i] = adj_matrix[:,3 + i, 0] = 2  
            adj_matrix[:,1, 3 + K + i] = adj_matrix[:,3 + K + i, 1] = 2  
            adj_matrix[:,2, 3 + 2*K + i] = adj_matrix[:,3 + 2*K + i, 2] = 2 
            
        return adj_matrix
    

class Direct_Emotion_Extended_Graph(nn.Module):
    "Direct_Emotion_Extended_Graph"
    
    def __init__(self, hyp_params):
        super(Direct_Emotion_Extended_Graph, self).__init__()
        
        self.dim = hyp_params['orig_d_l']

        self.RGAT = RGAT(nfeat=self.dim, nhid=self.dim, num_relation=4)
        self.linear_scene_audios = nn.Linear(1024, 256)
        self.linear_face_audios = nn.Linear(1024, 256)
        self.linear_text_audios = nn.Linear(1024, 256)

    def forward(self, 
                graph_stage_2_output=None,
                scene_ref_aduios=None, 
                face_ref_aduios=None, 
                text_ref_aduios=None):
                

        scene_ref_audios_256 = self.linear_scene_audios(scene_ref_aduios)
        face_ref_audios_256 = self.linear_face_audios(face_ref_aduios)
        text_ref_audios_256 = self.linear_text_audios(text_ref_aduios)
        

        graph_nodes = torch.cat([graph_stage_2_output, scene_ref_audios_256, face_ref_audios_256, text_ref_audios_256], dim=1)

   
        adj_matrix = self.build_adjacency_matrix(scene_ref_audios_256)


        
        Graph_Stage_3_output = self.RGAT(graph_nodes, adj_matrix)       
        
        return Graph_Stage_3_output
    
    def build_adjacency_matrix(self,scene_ref_audios_256):


        K = scene_ref_audios_256.size(1)  
        batch_size = scene_ref_audios_256.size(0)

        total_nodes = 3 + 3 * K + 3 * K
        adj_matrix = torch.zeros((batch_size, total_nodes, total_nodes), dtype=torch.long, device=scene_ref_audios_256.device)
        
   
        adj_matrix[:,0, 1] = adj_matrix[:,1, 0] = 1  
        adj_matrix[:,0, 2] = adj_matrix[:,2, 0] = 1  
        adj_matrix[:,1, 2] = adj_matrix[:,2, 1] = 1 

  

        for i in range(K):
            adj_matrix[:,0, 3 + i] = adj_matrix[:,3 + i, 0] = 2  
            adj_matrix[:,1, 3 + K + i] = adj_matrix[:,3 + K + i, 1] = 2 
            adj_matrix[:,2, 3 + 2*K + i] = adj_matrix[:,3 + 2*K + i, 2] = 2  
            
            adj_matrix[:,0, 3 + i + 3*K] = adj_matrix[:,3 + i + 3*K, 0] = 3
            adj_matrix[:,1, 3 + K + i + 3*K] = adj_matrix[:,3 + K + i + 3*K, 1] = 3
            adj_matrix[:,2, 3 + 2*K + i + 3*K] = adj_matrix[:,3 + 2*K + i + 3*K, 2] = 3
            
            
        return adj_matrix
    



    

class Aggregator(nn.Module):
    "Aggregator"
    
    def __init__(self, hyp_params, ):
        super(Aggregator, self).__init__()
        
        self.dim = hyp_params['orig_d_l']


        self.cross_att = nn.MultiheadAttention(self.dim, 1, dropout=0.1)
        self.pro_output = nn.Conv1d(self.dim * 2, self.dim, kernel_size=1, padding=0, bias=False)      
        

    def forward(self, 
                text_feature = None,
                graph_feature = None,
                graph_masks = None,
                ):
        
        cross_output, _ = self.cross_att(
            query = text_feature.transpose(0, 1),
            key = graph_feature.transpose(0, 1),
            value = graph_feature.transpose(0, 1),
            key_padding_mask = graph_masks
        )
        
        cross_output = cross_output.transpose(0, 1)        
        output = torch.cat([text_feature, cross_output], dim=-1)
        output = self.pro_output(output.transpose(1, 2)).transpose(1, 2)
  
        return output



class Style_dubber_model_15_SPost_Duration(nn.Module):
    """ Authentic-Dubber based on style dubber """
    def __init__(self, preprocess_config, model_config, train_config, hyp_params):
        super(Style_dubber_model_15_SPost_Duration, self).__init__()
        self.model_config = model_config

        self.variance_adaptor = VarianceAdaptor_softplus1(preprocess_config, model_config)
        self.decoder_Condition = Decoder_Condition(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],)
        self.postnet = PostNet() # Style, mel
        self.lip_encoder = Lip_Encoder(model_config)
        # we set the "is_tunable = False", it equals traditional multi-head attention. See https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention
        self.attn_lip_text  = StepwiseMonotonicMultiheadAttention(256, 256//8, 256//8)  # multihead == 8
        

        
        self.Basic_Emotion_Graph = Basic_Emotion_Graph(hyp_params)
        self.Indirect_Emotion_Extended_Graph = Indirect_Emotion_Extended_Graph(hyp_params)
        self.Direct_Emotion_Extended_Graph = Direct_Emotion_Extended_Graph(hyp_params)
        
        self.aggregator_stage_1 = Aggregator(hyp_params)
        self.aggregator_stage_2 = Aggregator(hyp_params)
        self.aggregator_stage_3 = Aggregator(hyp_params)
        
        self.emoID_classifier = AdversarialClassifier(
            in_dim=model_config["downsample_encoder"]["out_dim"],
            out_dim=8,
            hidden_dims=model_config["classifier"]["cls_hidden"]
        )        
        
        self.proj_fusion = nn.Conv1d(512, 256, kernel_size=1, padding=0, bias=False)      
        
        
    def parse_batch(self, batch):
        id_basename = batch["id"]
        emotion_ids = torch.from_numpy(batch["emotion_ids"]).long().cuda()
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()        
        scene_refs = torch.from_numpy(batch["scene_refs"]).float().cuda()
        face_refs = torch.from_numpy(batch["face_refs"]).float().cuda()
        text_refs = torch.from_numpy(batch["text_refs"]).float().cuda()
        scene_ref_audios = torch.from_numpy(batch["scene_ref_audios"]).float().cuda()
        face_ref_audios = torch.from_numpy(batch["face_ref_audios"]).float().cuda()
        text_ref_audios = torch.from_numpy(batch["text_ref_audios"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        scene_emo_embedding = torch.from_numpy(batch["scene_emo_embedding"]).float().cuda()
        face_emo_embedding = torch.from_numpy(batch["face_emo_embedding"]).float().cuda()
        text_emo_embedding = torch.from_numpy(batch["text_emo_embedding"]).float().cuda()
        emos_embedding = torch.from_numpy(batch["emos_embedding"]).float().cuda()
        return id_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding, emos_embedding, emotion_ids, scene_emo_embedding, face_emo_embedding, text_emo_embedding, scene_refs, face_refs, text_refs, scene_ref_audios, face_ref_audios, text_ref_audios

    def parse_batch_Setting3(self, batch):
        id_basename = batch["id"]
        zeroref_basename = batch["zerorefs"]
        emotion_ids = torch.from_numpy(batch["emotion_ids"]).long().cuda()
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        emos_embedding = torch.from_numpy(batch["emos_embedding"]).float().cuda()
        return id_basename, zeroref_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding, emos_embedding, emotion_ids

    def parse_batch_GRID(self, batch):
        id_basename = batch["id"]
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        return id_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding


    def parse_batch_Setting3_GRID(self, batch):
        id_basename = batch["id"]
        zeroref_basename = batch["zerorefs"]
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        return id_basename, zeroref_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding

    def forward(
        self,
        output, 
        text_encoder,
        src_masks,
        ref_mels,
        ref_mel_lens,
        face_lens,
        max_face_lens,
        lip_embedding,
        spk_embedding,
        mel_lens=None,
        max_mel_len=None,
        d_targets=None,
        p_targets=None,
        e_targets=None,
        mel_target=None,
        infer_flag=False,
        scene_emo_embedding =None,
        face_emo_embedding =None, 
        text_emo_embedding =None, 
        scene_refs =None, 
        face_refs =None, 
        text_refs =None, 
        scene_ref_audios =None, 
        face_ref_audios =None, 
        text_ref_audios =None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):  
        



        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        
        mel_target_masks = mel_masks
        
        max_ref_mel_lens = ref_mels.shape[1]
        ref_mel_masks = get_mask_from_lengths(ref_mel_lens, max_ref_mel_lens)

        lip_masks = get_mask_from_lengths(face_lens, max_face_lens)
        lip_embedding = self.lip_encoder(lip_embedding, lip_masks)
        
        if src_masks is not None:
            slf_attn_mask_text = src_masks.unsqueeze(1).expand(-1, max_face_lens, -1)
            slf_attn_mask_lip = lip_masks.unsqueeze(1).expand(-1, src_masks.size(1), -1)
            slf_attn_mask = slf_attn_mask_text.transpose(1,2) | slf_attn_mask_lip
        
        output_text_lip, AV_attn, _ =self.attn_lip_text(text_encoder, lip_embedding, lip_embedding, face_lens, mask=slf_attn_mask, query_mask=src_masks.unsqueeze(2))
        
        (
            output,
            log_d_predictions,
            d_rounded_pred,
            mel_lens_pred,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            output_text_lip,
            src_masks,
            mel_masks,
            max_mel_len,
            mel_lens,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        
        Basic_Emotion_Graph_output = self.Basic_Emotion_Graph(scene_emo_embedding, face_emo_embedding, text_emo_embedding)

        
        batch_size = Basic_Emotion_Graph_output.size(0)
        
        Basic_Emotion_Graph_output_lens = torch.tensor([3] * batch_size, dtype=torch.long, device=Basic_Emotion_Graph_output.device)
        Basic_Emotion_Graph_output_mask = get_mask_from_lengths(Basic_Emotion_Graph_output_lens, 3)
        
        aggregator_stage_1_output = self.aggregator_stage_1(output, Basic_Emotion_Graph_output, Basic_Emotion_Graph_output_mask)
        
        Indirect_Emotion_Extended_Graph_output = self.Indirect_Emotion_Extended_Graph(Basic_Emotion_Graph_output, scene_refs, face_refs, text_refs)
        

        Indirect_Emotion_Extended_Graph_output_len = Indirect_Emotion_Extended_Graph_output.size(1)
        
        Indirect_Emotion_Extended_Graph_output_lens = torch.tensor([Indirect_Emotion_Extended_Graph_output_len] * batch_size, dtype=torch.long, device=Indirect_Emotion_Extended_Graph_output.device)
        Indirect_Emotion_Extended_Graph_output_mask = get_mask_from_lengths(Indirect_Emotion_Extended_Graph_output_lens, max(Indirect_Emotion_Extended_Graph_output_lens))
        
        aggregator_stage_2_output = self.aggregator_stage_2(aggregator_stage_1_output, Indirect_Emotion_Extended_Graph_output, Indirect_Emotion_Extended_Graph_output_mask)

        
        Direct_Emotion_Extended_Graph_output = self.Direct_Emotion_Extended_Graph(Indirect_Emotion_Extended_Graph_output, scene_ref_audios, face_ref_audios, text_ref_audios)
        

        Direct_Emotion_Extended_Graph_output_len = Direct_Emotion_Extended_Graph_output.size(1)
        
        Direct_Emotion_Extended_Graph_output_lens = torch.tensor([Direct_Emotion_Extended_Graph_output_len] * batch_size, dtype=torch.long, device=Direct_Emotion_Extended_Graph_output.device)
        Direct_Emotion_Extended_Graph_output_mask = get_mask_from_lengths(Direct_Emotion_Extended_Graph_output_lens, max(Direct_Emotion_Extended_Graph_output_lens))
        
        aggregator_stage_3_output = self.aggregator_stage_3(aggregator_stage_2_output, Direct_Emotion_Extended_Graph_output, Direct_Emotion_Extended_Graph_output_mask)
        
        
        emotion_embedding_masks = (1 - get_mask_from_lengths(mel_lens_pred, max(mel_lens_pred)).float()).unsqueeze(-1).expand(-1, -1, 256)
        emotion_id_embedding = torch.sum(aggregator_stage_3_output * emotion_embedding_masks, axis = 1) / mel_lens_pred.unsqueeze(-1).expand(-1, 256)
        emotion_id_embedding = self.emoID_classifier(emotion_id_embedding, is_reversal=False)

        
        fusion_output = torch.cat([output, aggregator_stage_3_output], dim=-1)
        fusion_output = self.proj_fusion(fusion_output.transpose(1, 2)).transpose(1, 2)
        
        output, mel_masks = self.decoder_Condition(fusion_output, mel_masks, spk_embedding)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output, spk_embedding) + output

        if d_targets is not None:
            return (
                output,
                postnet_output,
                log_d_predictions,
                d_rounded_pred,
                src_masks,
                mel_masks,
                mel_lens,
                ref_mel_masks,
                AV_attn,
                lip_masks,
                None,
                emotion_id_embedding
            )
        else:
            return postnet_output, mel_lens_pred


        mel_target = mel_target.permute(0, 2, 1)
        output = output.permute(0, 2, 1) 
        

         
        if output.shape[-1] % 2 != 0:

            output_input_decoder = F.pad(output, (0, 1), mode="constant", value=0)  
            mel_target_input_decoder = F.pad(mel_target, (0, 1), mode="constant", value=0)  
            padding_length = 1
            mel_lens_input_decoder = mel_lens + 1
            mel_masks_input_decoder = (
                get_mask_from_lengths(mel_lens_input_decoder, max_mel_len + 1)
                    if mel_lens is not None
                    else None)
            mel_masks_input_decoder =  mel_masks_input_decoder.unsqueeze(1)
            mel_masks_input_decoder = ~mel_masks_input_decoder
            if  infer_flag == False:
                diff_loss, _ = self.matchatts_decoder.compute_loss(x1=mel_target_input_decoder, mask=mel_masks_input_decoder, mu=output_input_decoder)
                prior_loss = torch.sum(0.5 * ((mel_target_input_decoder - output_input_decoder) ** 2 + math.log(2 * math.pi)) * mel_masks_input_decoder)
                prior_loss = prior_loss / (torch.sum(mel_masks_input_decoder) * 80)
                noise_loss = prior_loss + diff_loss

            mel_pred_masks_input_decoder = (
                get_mask_from_lengths(mel_lens_pred+1, max(mel_lens_pred) + 1)
                    if mel_lens_pred is not None
                    else None)
            mel_pred_masks_input_decoder =  mel_pred_masks_input_decoder.unsqueeze(1)
            mel_pred_masks_input_decoder = ~mel_pred_masks_input_decoder
            postnet_output = self.matchatts_decoder(output_input_decoder, mel_pred_masks_input_decoder, 50) 
            postnet_output = postnet_output[:, :, :-padding_length]  
            
                           
        else:
            output_input_decoder = output
            mel_target_input_decoder = mel_target
            mel_masks_input_decoder = (
                get_mask_from_lengths(mel_lens, max_mel_len)
                    if mel_lens is not None
                    else None)
            mel_masks_input_decoder =  mel_masks_input_decoder.unsqueeze(1)
            mel_masks_input_decoder = ~mel_masks_input_decoder
            if  infer_flag == False:
                diff_loss, _ = self.matchatts_decoder.compute_loss(x1=mel_target_input_decoder, mask=mel_masks_input_decoder, mu=output_input_decoder)
                prior_loss = torch.sum(0.5 * ((mel_target_input_decoder - output_input_decoder) ** 2 + math.log(2 * math.pi)) * mel_masks_input_decoder)
                prior_loss = prior_loss / (torch.sum(mel_masks_input_decoder) * 80)
                noise_loss = prior_loss + diff_loss
            
            mel_pred_masks_input_decoder = (
                get_mask_from_lengths(mel_lens_pred, max(mel_lens_pred))
                    if mel_lens_pred is not None
                    else None)
            mel_pred_masks_input_decoder =  mel_pred_masks_input_decoder.unsqueeze(1)
            mel_pred_masks_input_decoder = ~mel_pred_masks_input_decoder
            postnet_output = self.matchatts_decoder(output_input_decoder, mel_pred_masks_input_decoder, 50)
            
            
        postnet_output = postnet_output.permute(0, 2, 1)  
        output = output.permute(0, 2, 1)    
        
   

        if d_targets is not None:
            return (
                output,
                postnet_output,
                log_d_predictions,
                d_rounded_pred,
                src_masks,
                mel_masks,
                mel_lens,
                ref_mel_masks,
                AV_attn,
                lip_masks,
                noise_loss
            )
        else:
            return postnet_output, mel_lens_pred
        

