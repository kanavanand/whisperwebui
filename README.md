## Whisper 

Whisper is an automatic speech recognition model trained on 680,000 hours of multilingual data collected from the web. As per OpenAI, this model is robust to accents, background noise and technical language. In addition, it supports 99 different languages’ transcription and translation from those languages into English.. In this Daisi I have used the "small" model, but soon I'll be updating the other 4 variants of this model as well.

###### Link  - https://app.daisi.io/daisies/kanav/Whisper%20Model-GPU/api
##### Use daisi-api to make calls

Step1- Load model
```
import pydaisi as pyd
whisper_model_gpu = pyd.Daisi("kanav/Whisper Model-GPU")
```
Step2- Make predictions <br>
Predictions can be made sending byte audio object or audio array (in float format )
From audio array(single dimensional)
```
whisper_model_gpu.inference(audio).value
```
From byte like object

```
whisper_model_gpu.infer_wave_byte(wave_bytes).value
```