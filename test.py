import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

class DoubleConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)
    

class Up(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 転置畳み込みでは、チャネル数は 1/2 に、画像サイズ h, w は 2倍
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        '''
        x1 : Decoderからの出力（下から）
        x2 : Encoderからの出力（左から）
        '''
        x1 = self.up(x1)

        # x1 と x2 の大きさのズレを測る
        # x1 の形状は (batch_size, c, h, w)
        # x1.size()[2] は height を取得
        # x1.size()[3] は width を取得
        diff_h = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_w = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 を zero padding して x2 の h, w と揃える
        x1 = F.pad(x1, [diff_h // 2, diff_h - diff_h // 2,
                            diff_w // 2, diff_w - diff_w // 2])

        # channel 方向に結合
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
    


class Net(pl.LightningModule):
    def __init__(self, in_channels, hidden_size, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.inconv = DoubleConv(in_channels, hidden_size)

        self.down1 = Down(hidden_size, hidden_size*2)
        self.down2 = Down(hidden_size*2, hidden_size*4)
        self.down3 = Down(hidden_size*4, hidden_size*8)
        self.down4 = Down(hidden_size*8, hidden_size*8)

        self.up1 = Up(hidden_size*16, hidden_size*4)
        self.up2 = Up(hidden_size*8, hidden_size*2)
        self.up3 = Up(hidden_size*4, hidden_size)
        self.up4 = Up(hidden_size*2, hidden_size)

        self.outconv = nn.Conv2d(hidden_size, n_classes, kernel_size=1)

    def forward(self, x):
        '''
        Encoder の処理
        '''
        x1 = self.inconv(x) # ch:3   -> 64,  size: 224 * 224  ->  224 * 224
        x2 = self.down1(x1) # ch:64  -> 128, size: 224 * 224  ->  112 * 112
        x3 = self.down2(x2) # ch:128 -> 256, size: 112 * 112  ->  56 * 56
        x4 = self.down3(x3) # ch:256 -> 512, size: 56 * 56    ->  28 * 28
        x5 = self.down4(x4) # ch:512 -> 512, size: 28 * 28    ->  14 * 14

        '''
        Decoder の処理
        '''
        x = self.up1(x5, x4) # ch:512 -> 256, size: 14 * 14   -> 28, 28
        x = self.up2(x, x3)  # ch:256 -> 128, size: 28 * 28   -> 56, 56
        x = self.up3(x, x2)  # ch:128 -> 64,  size: 56 * 56   -> 112, 112
        x = self.up4(x, x1)  # ch: 64 -> 64, size: 112 * 112 -> 224 * 224

        return self.outconv(x) # ch:64 -> 12, size: 224 * 224 -> 224, 224


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy(y.softmax(dim=1), t, task='multiclass', num_classes = 2), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=1), t, task='multiclass', num_classes = 2), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer



# 学習済みモデルの読み込み
loaded_model = torch.load('unet30_16.pth', map_location=torch.device('cpu'))
model_state_dict = loaded_model  # state_dictを取得

# モデルの定義
model = Net(in_channels=1, hidden_size=32, n_classes=2)  # モデルの定義と初期化
model.load_state_dict(model_state_dict)  # state_dictをモデルにロード
model.eval()  # モデルを評価モードに設定

# 画像を推論する関数
def inference(image):
    # 画像の前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # 推論
    with torch.no_grad():
        output = model(image)

    # 推論結果を画像に変換
    output = torch.argmax(output, dim=1)
    output = output.squeeze(0).detach().cpu().numpy()

    return output

# IoUの計算
def calculate_iou(pred, target):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Streamlitアプリケーションの設定
st.title('画像推論アプリ')
st.write('ここに画像ファイルをドロップしてください')

uploaded_file = st.file_uploader('画像を選択', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 画像を読み込み、推論して結果を表示
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)

    output = inference(image)
    st.image(output, caption='推論結果', use_column_width=True)

    # IoUの計算と表示
    target = np.array(image)
    iou_score = calculate_iou(output, target)
    st.write(f'IoUスコア: {iou_score:.4f}')
