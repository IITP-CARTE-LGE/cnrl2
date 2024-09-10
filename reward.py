import torch
import torch.nn as nn
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoImageProcessor, AutoModel
from bert_score import BERTScorer
from controlnet_aux import OpenposeDetector
from tqdm import tqdm
import numpy as np


class AlignmentScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device=device
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", device_map=self.device)
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", device_map=self.device, torch_dtype=torch.float16, quantization_config=self.quantization_config,low_cpu_mem_usage=True)
        self.scorer = BERTScorer(model_type='bert-base-uncased', device=self.device)

    @torch.no_grad()
    def __call__(self, images, prompts):
        prompt='[INST] <image>\nWhat is shown in this image? [/INST]'
        questions=[prompt]*len(images)
        inputs = self.processor(text=questions, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device).long() for k, v in inputs.items()}

        # Generate
        generate_ids  = self.model.generate(**inputs, max_length=40)
        outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs = list(map(lambda l: l.split('[/INST]')[-1].lower().strip(), outputs))
        P, R, F1 = self.scorer.score(outputs, list(prompts))
        return R.to(self.device)

class PoseScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(self.device)
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', device_map=self.device)
        self.model = AutoModel.from_pretrained('facebook/dinov2-base', device_map=self.device)

    def feature_extraction(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        image_features = outputs.last_hidden_state
        image_features = image_features.mean(dim=1)
        return image_features

    @torch.no_grad()
    def __call__(self, original_condition_images, generated_images, prompts, global_step, image_log_dir):
        condition_images=[]
        no_pose_list = []
        
        for i, image in enumerate(generated_images):
            image, count = self.openpose(image, detect_resolution=1024, image_resolution=1024, hand_and_face=False)
            if count == 0:
                no_pose_list.append(i)
            condition_images.append(image)

        if '0' not in no_pose_list and global_step%100==0:
            generated_images[0].save(f'{image_log_dir}/{global_step}_generated_0.png')
            condition_images[0].save(f'{image_log_dir}/{global_step}_extracted_0.png')
            original_condition_images[0].save(f'{image_log_dir}/{global_step}_original_0.png')
        
        original_condition_features = self.feature_extraction(original_condition_images)
        condition_features = self.feature_extraction(condition_images)
        cos = nn.CosineSimilarity(dim=1)
        sim = cos(original_condition_features, condition_features)
        #min max scaling
        min_, max_ = 0.9, 1
        sim=(sim-min_)/(max_-min_)
        if len(no_pose_list) != 0:
            for idx in no_pose_list:
                sim[idx] = torch.tensor(-1, dtype=sim[idx].dtype)
        return sim

class PoseKeypointScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(self.device)
        self.body_parts_name = ['body', 'left_hand', 'right_hand', 'face']


    def cosine_distance(self, pose1, pose2):
        total_sim=[]
        for body_part in self.body_parts_name:
            if pose1[body_part] is None and pose2[body_part] is None:
                continue
            if pose1[body_part] is None or pose2[body_part] is None:
                if pose1[body_part] is None:
                    pose1[body_part] = [-1]*len(pose2[body_part])
                else:
                    pose2[body_part] = [-1]*len(pose1[body_part])

            p1 = np.array(pose1[body_part])
            p2 = np.array(pose2[body_part])
            cossim = p1.dot(np.transpose(p2)) / (np.linalg.norm(p1) * np.linalg.norm(p2))
            cossim = cossim.astype('float16') 

            total_sim.append(cossim)
        return round(sum(total_sim)/len(total_sim),6)

    def prepare_keypoints(self, keypoints, w=1024, h=1024):
        body_parts = {'body':[],'left_hand':[], 'right_hand':[], 'face':[]}

        keypoints_list = []
        for keypoint in keypoints[0].body.keypoints:  
            if keypoint is not None:
                x, y = keypoint.x, keypoint.y
                x = int(x * w)
                y = int(y * h)
                keypoints_list.append(x)
                keypoints_list.append(y)
            else:
                keypoints_list.append(-1)
                keypoints_list.append(-1)
        body_parts['body'] = keypoints_list

        for i, keypoints in enumerate(keypoints[0][1:]):  
            keypoints_list = []
            if keypoints is not None:
                for keypoint in keypoints:
                    x, y = keypoint.x, keypoint.y
                    if x < 0 and y < 0:
                        x, y = -1, -1
                    else:
                        x = float(x * w)
                        y = float(y * h)
                    keypoints_list.append(round(x, 6))
                    keypoints_list.append(round(y, 6))
            else:
                keypoints_list = None
            body_parts[self.body_parts_name[i+1]] = keypoints_list

        return body_parts

    @torch.no_grad()
    def __call__(self, original_condition_images, original_images, generated_images, prompts, global_step, image_log_dir):
        gen_conds=[]
        org_conds=[]
        scores = []
        
        for i in range(len(generated_images)):
            gen_cond, gen_keypoints = self.openpose(generated_images[i], detect_resolution=1024, image_resolution=1024, return_keypoints=True, hand_and_face=True)
            org_cond, org_keypoints = self.openpose(original_images[i], detect_resolution=1024, image_resolution=1024, return_keypoints=True, hand_and_face=True)
            gen_conds.append(gen_cond)
            org_conds.append(org_cond)
            # if more than one human is detected, score = 0
            if len(gen_keypoints) == 1 and len(org_keypoints) == 1: 
                gen_keypoints = self.prepare_keypoints(gen_keypoints)
                org_keypoints = self.prepare_keypoints(org_keypoints)
                score = self.cosine_distance(gen_keypoints, org_keypoints)
            else:
                score = 0
            scores.append(score)
            
        generated_images[0].save(f'{image_log_dir}/{global_step}_gen_0.png')
        gen_conds[0].save(f'{image_log_dir}/{global_step}_gen_cond_0.png')
        org_conds[0].save(f'{image_log_dir}/{global_step}_org_cond_0.png')
        # original_images[0].save(f'{image_log_dir}/{global_step}_org_0.png')
        return torch.as_tensor(scores, device=self.device, dtype=torch.float16)


class RewardComputation():
    def __init__(self, device, reward_type):
        self.device=device
        self.alignment_scorer = AlignmentScorer(self.device)
        if reward_type == 'feature':
            self.pose_scorer = PoseScorer(self.device)
        elif reward_type == 'keypoints':
            self.pose_scorer = PoseKeypointScorer(self.device)
        

    def reward_fn(self, condition_images, images, prompts, metadata, global_step, image_log_dir):
        alignment_scores = self.alignment_scorer(images, prompts)
        pose_scores = self.pose_scorer(condition_images, images, prompts, global_step, image_log_dir)
        scores = alignment_scores + pose_scores #temporary expedient

        return scores, {}

    def compute_rewards(self, prompt_image_pairs, global_step, image_log_dir):
        rewards = []
        for original_images, condition_images, generated_images, prompts, prompt_metadata in prompt_image_pairs:
            reward, reward_metadata = self.reward_fn(condition_images, generated_images, prompts, prompt_metadata, global_step, image_log_dir)
            rewards.append(
                (
                    torch.as_tensor(reward, device=self.device),
                    reward_metadata,
                )
            )
        return zip(*rewards)

    def reward_fn2(self, original_images, condition_images, generated_images, prompts, prompt_metadata, global_step, image_log_dir):
        alignment_scores = self.alignment_scorer(generated_images, prompts)
        pose_scores = self.pose_scorer(condition_images, original_images, generated_images, prompts, global_step, image_log_dir)
        scores = alignment_scores + pose_scores #temporary expedient
        return scores, {}

    def compute_rewards2(self, prompt_image_pairs, global_step, image_log_dir):
        rewards = []
        for original_images, condition_images, generated_images, prompts, prompt_metadata in prompt_image_pairs:
            reward, reward_metadata = self.reward_fn2(original_images, condition_images, generated_images, prompts, prompt_metadata, global_step, image_log_dir)
            rewards.append(
                (
                    torch.as_tensor(reward, device=self.device),
                    reward_metadata,
                )
            )
        return zip(*rewards)

