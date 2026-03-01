cd models
git clone --recursive https://github.com/ByteDance/Hi3DGen.git
cd Hi3DGen
# pytorch
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu121==2.3.6 xformers==0.0.27.post2
# other dependencies
pip install -r requirements.txt
pip install nibabel