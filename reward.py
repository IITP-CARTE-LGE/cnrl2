import torch
import torch.nn as nn
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoImageProcessor, AutoModel
from bert_score import BERTScorer
from controlnet_aux import OpenposeDetector
from tqdm import tqdm


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
    def __call__(self, images, prompts):
        condition_images=[]
        for image in images:
            image = self.openpose(image)
            condition_images.append(image)
        
        image_features = self.feature_extraction(images)
        condition_features = self.feature_extraction(condition_images)
        cos = nn.CosineSimilarity(dim=1)
        sim = cos(image_features, condition_features)
        return sim

class RewardComputation():
    def __init__(self, device):
        self.device=device
        self.alignment_scorer = AlignmentScorer(self.device)
        self.pose_scorer = PoseScorer(self.device)

    def reward_fn(self, images, prompts, metadata):
        alignment_scores = self.alignment_scorer(images, prompts)
        pose_scores = self.pose_scorer(images, prompts)
        scores = alignment_scores + pose_scores
        return scores, {}

    def compute_rewards(self, prompt_image_pairs):
        rewards = []
        for images, prompts, prompt_metadata in prompt_image_pairs:
            reward, reward_metadata = self.reward_fn(images, prompts, prompt_metadata)
            rewards.append(
                (
                    torch.as_tensor(reward, device=self.device),
                    reward_metadata,
                )
            )
        return zip(*rewards)

if __name__=="__main__":
    from PIL import Image
    import torch.nn as nn
    image_list = ['/scratch/heywon/data/openpose/0a0d2e231f162f2d.png','/scratch/heywon/data/openpose/0a4a0b2f2b2f1b0b.png','/scratch/heywon/data/openpose/0a4b0f27232f1f2b.png']
    images = []
    for image in image_list:
        images.append(Image.open(image))
    pose_scorer = PoseScorer('cuda')
    ch = pose_scorer.openpose(images[0])
    print(ch)

    feature1 = pose_scorer.feature_extraction(images[0])
    feature2 = pose_scorer.feature_extraction(images[1])
    feature3 = pose_scorer.feature_extraction(images[2])

    cos = nn.CosineSimilarity(dim=1)
    sim1 = cos(feature1, feature2)
    sim2 = cos(feature2, feature3)
    sim3 = cos(feature1, feature3)
    print(sim1,sim2,sim3)