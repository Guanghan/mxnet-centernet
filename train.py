from models.model import create_model, load_model, save_model
from opts import opts

print('Creating model...')
opt = opts().init()
model = create_model(opt.arch, opt.heads, opt.head_conv)

print("Saving model...")
save_model(model, "./init_params.params")
