from clip_tokenizer.clip_lang_encoder import LangClip



model = LangClip(freeze_backbone=True, model_name="ViT-B/32")

lang = "use the mixer"

for i in range(5):
    lang_emb = model(lang).squeeze()
    print(lang_emb[:20])

# output example
# tensor([-0.0189,  0.1305,  0.0655, -0.1547, -0.1909, -0.1614, -0.0601, -1.2949,
#          0.0541,  0.3203, -0.4299,  0.0810,  0.4946,  0.1282,  0.1354,  0.0926,
#          0.1227,  0.0378, -0.1241, -0.1549], device='cuda:0',
#        dtype=torch.float16)