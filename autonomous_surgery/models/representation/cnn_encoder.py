import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicRepresentationEncoder(nn.Module):
    """
    Extremely simple multimodal encoder.

    Inputs
    ------
    endoscope_image : [B,3,H,W]
    wrist_l         : [B,3,H,W]
    wrist_r         : [B,3,H,W]
    robot_state     : [B,S]

    Outputs
    -------
    global_token : [B,D]
    tokens       : [B,N,D]
    """

    def __init__(
        self,
        robot_state_dim: int,
        model_emb_dim: int = 512,
    ):
        super().__init__()

        self.model_emb_dim = model_emb_dim

        # ------------------------------------------------
        # Image encoders
        # ------------------------------------------------

        def make_cnn():
            return nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        self.endoscope_cnn = make_cnn()
        self.wrist_l_cnn = make_cnn()
        self.wrist_r_cnn = make_cnn()

        self.image_proj = nn.Linear(128, model_emb_dim)

        # ------------------------------------------------
        # Robot state encoder
        # ------------------------------------------------

        self.state_mlp = nn.Sequential(
            nn.Linear(robot_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, model_emb_dim),
        )

        # ------------------------------------------------
        # Global token
        # ------------------------------------------------

        self.global_proj = nn.Linear(model_emb_dim * 4, model_emb_dim)

    def encode_image(self, cnn, img):

        x = cnn(img)           # [B,128,1,1]
        x = x.flatten(1)       # [B,128]
        x = self.image_proj(x) # [B,D]

        return x

    def forward(
        self,
        endoscope_image,
        wrist_l,
        wrist_r,
        robot_states,
        texts=None,
    ):

        B = robot_states.shape[0]

        # ------------------------------------------------
        # Encode images
        # ------------------------------------------------

        endo_feat = self.encode_image(self.endoscope_cnn, endoscope_image)
        wl_feat = self.encode_image(self.wrist_l_cnn, wrist_l)
        wr_feat = self.encode_image(self.wrist_r_cnn, wrist_r)

        # ------------------------------------------------
        # Encode robot state
        # ------------------------------------------------

        state_feat = self.state_mlp(robot_states)

        # ------------------------------------------------
        # Tokens
        # ------------------------------------------------

        tokens = torch.stack(
            [endo_feat, wl_feat, wr_feat, state_feat],
            dim=1,
        )  # [B,4,D]

        # ------------------------------------------------
        # Global token
        # ------------------------------------------------

        global_input = torch.cat(
            [endo_feat, wl_feat, wr_feat, state_feat],
            dim=-1,
        )

        global_token = self.global_proj(global_input)

        return global_token, tokens