import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")



# inputs = processor(
#     text=["a sad slow piano melody"],
#     padding=True,
#     return_tensors="pt",
# )

# audio_values = model.generate(**inputs, max_new_tokens=256)

# sampling_rate = model.config.audio_encoder.sampling_rate
# scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())




# inputs = processor(
#     text=[
#         "a sad slow piano melody in minor key",
#         "a happy upbeat electronic dance music",
#         "a calm relaxing ambient music with soft strings",
#         "an anxious tense orchestral piece with fast tempo",
#     ],
#     padding=True,
#     return_tensors="pt",
# )

# audio_values = model.generate(**inputs, max_new_tokens=256)

# sampling_rate = model.config.audio_encoder.sampling_rate
# emotions = ["sad", "happy", "calm", "anxious"]
# for i, name in enumerate(emotions):
#     scipy.io.wavfile.write(f"{name}_music.wav", rate=sampling_rate, data=audio_values[i, 0].numpy())
#     print(f"{name}_music.wav 已生成")
    

# 看看文本经过processor后变成了什么
inputs = processor(text=["a sad piano melody"], padding=True, return_tensors="pt")
print(inputs.keys())           # 看有哪些字段
print(inputs['input_ids'].shape)  # 看文本编码的维度

# 看看模型内部结构
print(model)  # 打印模型结构，找到条件注入的位置

print("done!")