import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

try:
    from visionstream.codecs.base import BaseIntraCodec
    from visionstream.registry import register_codec
except ImportError:
    # Fallback/mock for standalone execution or when framework is not initialized
    class BaseIntraCodec(nn.Module):
        pass
    def register_codec(name, cls):
        pass

class BaseFeatureAutoEncoder(BaseIntraCodec, nn.Module):
    """
    확장 가능한 기초 Feature AutoEncoder 모듈입니다.
    사용자는 이 클래스를 상속받아 encoder, decoder, entropy_bottleneck을 자신의 모듈로 교체 및 구현할 수 있습니다.
    Web UI와의 연동을 고려하여 각 블록(인코딩, 양자화/엔트로피, 디코딩)이 독립적으로 분리되어 있습니다.
    """
    def __init__(self, in_channels: int = 256, latent_channels: int = 64, device: str = "cpu", **kwargs):
        nn.Module.__init__(self)
        try:
            BaseIntraCodec.__init__(self) # Initialize both Base classes if framework available
        except Exception:
            pass
            
        self.device = device
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # 1. 인코더 블록
        self.encoder = self.build_encoder()
        
        # 2. 패킷 양자화 및 엔트로피 (더미)
        self.entropy_bottleneck = self.build_entropy_bottleneck()
        
        # 3. 디코더 블록
        self.decoder = self.build_decoder()
        
        self.to(device)
        
    def build_encoder(self) -> nn.Module:
        """사용자가 오버라이드 가능한 Encoder 정의. 본 예제에서는 공간 해상도를 유지하며 채널수를 줄이는 단순 1x1 Conv를 제공합니다."""
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.latent_channels, kernel_size=1)
        )
        
    def build_entropy_bottleneck(self) -> nn.Module:
        """사용자가 오버라이드 가능한 Entropy Bottleneck 정의. 엔트로피 확률 추정 모델 등을 포함합니다."""
        class DummyEntropy(nn.Module):
            def forward(self, x):
                # 실제 논문 구현체에서는 반올림 학습을 위해 gumbel softmax 또는 additive uniform noise를 사용합니다.
                y_hat = torch.round(x)
                # 확률 예측 더미 (Rate Loss 계산용)
                likelihoods = torch.ones_like(y_hat) * 0.5 
                return y_hat, likelihoods
        return DummyEntropy()

    def build_decoder(self) -> nn.Module:
        """사용자가 오버라이드 가능한 Decoder 정의. 축소된 Feature를 다시 원래 공간/채널 차원으로 복원합니다."""
        return nn.Sequential(
            nn.Conv2d(self.latent_channels, self.in_channels, kernel_size=1)
        )
        
    def loss_function(self, x: torch.Tensor, x_hat: torch.Tensor, likelihoods: torch.Tensor, task_loss: torch.Tensor = 0.0) -> Dict[str, torch.Tensor]:
        """
        R-D 커브 최적화 및 Task Aware 백프롭을 위한 종합 Loss 함수 서식.
        Web UI에서 사용자가 Rate vs Task Loss 파라미터를 시각적으로 조정할 수 있도록 각 항을 Dictionary로 반환합니다.
        """
        # BPP 계산 (더미)
        # 4D 텐서 기준 총 픽셀 (또는 Feature Element) 개수로 정규화
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3] if len(x.shape) == 4 else 1
        rate_loss = torch.log2(likelihoods).sum() / (-num_pixels)
        
        # Distortion 계산 (기본적으로 L2 Loss 사용, 필요 시 MSE/SSIM 복합 가능)
        dist_loss = nn.functional.mse_loss(x, x_hat)
        
        # 종합 Loss: (Rate Penalty) + (Distortion) + (목표 Task Loss)
        # 람다 계수(0.01 등)는 외부에서 주입하거나 스케쥴러를 통해 동적으로 변환할 수 있도록 확장합니다.
        total_loss = 0.01 * rate_loss + dist_loss + task_loss
        
        return {
            "total_loss": total_loss,
            "rate_loss": rate_loss,
            "dist_loss": dist_loss,
            "task_loss": task_loss
        }

    # BaseIntraCodec Method 구현 (실제 파이프라인 Inference 용도)
    def compress(self, x: torch.Tensor) -> bytes:
        """Feature Map Tensor 'x'를 압축하여 전송 가능한 byte 스트림을 반환합니다 (네트워크 병목 측정 등 활용)."""
        latent = self.encoder(x)
        y_hat, _ = self.entropy_bottleneck(latent)
        # 실제로는 Arithmetic Encoder(e.g., C++ Extension)를 통해 bitstream으로 직렬화
        return b"dummy_bitstream"

    def decompress(self, bitstream: bytes, shape: tuple = None) -> torch.Tensor:
        """Byte 스트림을 복원하여 Feature Tensor를 반환합니다."""
        # 실제로는 Arithmetic Decoder를 통해 y_hat을 복원
        # shape가 주어지지 않으면 내부 헤더를 디코드하여 알아내야 합니다 (구현체 별도)
        dummy_y_hat = torch.zeros((1, self.latent_channels, 32, 32), device=self.device)
        x_hat = self.decoder(dummy_y_hat)
        return x_hat

    # 훈련용 Forward 패스
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        End-to-End 연결 및 학습을 위한 순방향 채널 패스.
        비전 모델(예: YOLO 중간 레이어)의 출력을 받아 인코딩, 양자화, 디코딩을 수행합니다.
        """
        latent = self.encoder(x)
        y_hat, likelihoods = self.entropy_bottleneck(latent)
        x_hat = self.decoder(y_hat)
        
        # Web UI 등에서 시각화, 모니터링, 추론 결과를 다층적으로 분석하기 위해 정보를 반환합니다.
        info = {
            "latent": latent,
            "likelihoods": likelihoods,
            "bpp_estimate": float(torch.log2(likelihoods).sum() / - (x.shape[0]*x.shape[2]*x.shape[3])) if len(x.shape) == 4 else 0.0
        }
        return x_hat, info

# 범용 레지스트리 자동 등록
register_codec("feature_ae_dummy", BaseFeatureAutoEncoder)
