
import timm as timm


all_models = timm.list_models("*Unet*")
print(all_models)

# timm.create_model()
