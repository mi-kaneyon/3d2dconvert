import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, segmentation
from PIL import Image
import numpy as np

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前訓練済みモデルのロード
resnet_model = resnet50(pretrained=True).to(device)
resnet_model.eval()
segmentation_model = segmentation.deeplabv3_resnet50(pretrained=True).to(device)
segmentation_model.eval()

# 中間層の特徴量を抽出するフック
def get_intermediate_layer(layer_name):
    def hook(module, input, output):
        global features
        features = output
    return hook

layer_name = 'layer3'  # ResNetの任意の中間層名
layer = dict([*resnet_model.named_modules()])[layer_name]
layer.register_forward_hook(get_intermediate_layer(layer_name))

# 画像前処理のためのトランスフォーム
preprocess_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_segmentation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def apply_pseudo_3d_effect(frame):
    # フレームの前処理
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess_resnet(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 特徴量抽出
        resnet_model(input_tensor)

    # 特徴量を取得し、CPUに転送
    features_np = features.squeeze().cpu().numpy()
    
    # 特徴量をチャネル方向に平均化して2Dグレースケール画像に変換
    gray = np.mean(features_np, axis=0)
    
    # 特徴量を正規化
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    
    # 8ビットのグレースケール画像に変換
    gray = (gray * 255).astype(np.uint8)

    # グレースケール画像を元のフレームサイズにリサイズ
    gray_resized = cv2.resize(gray, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    return gray_resized

def segment_frame(frame):
    # フレームの前処理
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess_segmentation(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # セグメンテーション実行
        output = segmentation_model(input_tensor)['out']
    output_predictions = output.argmax(1).cpu().numpy()[0]

    return output_predictions

def draw_3d_segmentation(gray, seg_map):
    # グレースケール画像にヒートマップを適用
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # セグメンテーションマップのカラーマッピング
    unique_labels = np.unique(seg_map)
    color_map = np.zeros_like(heatmap)
    
    for label in unique_labels:
        if label == 0:  # 背景ラベルは無視
            continue
        mask = seg_map == label
        color_map[mask] = np.random.randint(0, 255, size=3)
    
    # 輪郭検出と描画
    contours, _ = cv2.findContours(seg_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_map, contours, -1, (0, 255, 0), 2)
    
    # 擬似3D効果とセグメンテーション結果のブレンド
    blended_frame = cv2.addWeighted(heatmap, 0.7, color_map, 0.3, 0)

    return blended_frame

def main():
    cap = cv2.VideoCapture(0)  # デバイスIDを0に設定

    if not cap.isOpened():
        print("カメラを開くことができません。デバイスIDを確認してください。")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("フレームを取得できません。")
            break
        
        # 擬似3D効果の適用
        gray_resized = apply_pseudo_3d_effect(frame)
        
        # セグメンテーション実行
        seg_map = segment_frame(frame)
        
        # 擬似3D効果とセグメンテーション結果を描画
        result_frame = draw_3d_segmentation(gray_resized, seg_map)
        
        # フレームを表示
        cv2.imshow('Pseudo 3D Effect with Segmentation', result_frame)
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
