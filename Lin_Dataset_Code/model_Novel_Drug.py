from mol import Mol
from hgnn import HGNN
from GraphLearner import GraphLearner
from RGCN import *

from subTree import subtreeExtractor
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
device = torch.device("cuda:0")
class SemanticRelationSmoother(nn.Module):
    def __init__(self, num_original_rels, num_semantic_rels, rel_embedding_dim, tau=0.1):
        super().__init__()
        self.num_original_rels = num_original_rels
        self.num_semantic_rels = num_semantic_rels
        self.tau = tau

        self.original_rel_embedding = nn.Embedding(num_original_rels, rel_embedding_dim)
        self.smoother = nn.Linear(rel_embedding_dim, num_semantic_rels)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.original_rel_embedding.weight)
        nn.init.xavier_uniform_(self.smoother.weight, gain=nn.init.calculate_gain('relu'))
        if self.smoother.bias is not None:
            nn.init.constant_(self.smoother.bias, 0)

    def get_type_mapping_distribution(self, hard=True):
        """返回 Gumbel-Softmax 采样分布"""
        unique_types = torch.arange(0, self.num_original_rels, device=self.original_rel_embedding.weight.device)
        rel_embeds = self.original_rel_embedding(unique_types)
        semantic_logits = self.smoother(rel_embeds)
        type_mapping = F.gumbel_softmax(semantic_logits, tau=self.tau, hard=hard, dim=-1)
        return type_mapping, semantic_logits

    def get_soft_distribution(self):

        unique_types = torch.arange(0, self.num_original_rels, device=self.original_rel_embedding.weight.device)
        rel_embeds = self.original_rel_embedding(unique_types)
        semantic_logits = self.smoother(rel_embeds)
        soft_dist = F.softmax(semantic_logits, dim=-1)
        return soft_dist

    def compute_kl_loss(self):
        soft_dist = self.get_soft_distribution()
        log_soft_dist = torch.log(soft_dist + 1e-12)

        target_dist = torch.full_like(soft_dist, 1.0 / self.num_semantic_rels)
        kl_loss = F.kl_div(log_soft_dist, target_dist, reduction="batchmean")
        return kl_loss

    def forward(self, original_edge_types):
        type_mapping, _ = self.get_type_mapping_distribution(hard=True)
        one_hot_edge_types = type_mapping[original_edge_types]
        new_edge_ids = torch.argmax(one_hot_edge_types, dim=-1)
        return new_edge_ids


class GNN1(nn.Module):
    def __init__(self, kg_g, args):
        super(GNN1, self).__init__()
        self.kg_g = kg_g.to(device)

        original_edge_types = self.kg_g.edata['edges']
        num_original_rels = original_edge_types.max().item() + 1
        self.num_semantic_rels = 64
        rel_embedding_dim = 128
        num_layer = 3
        dropout = 0.1

        self.smoother = SemanticRelationSmoother(num_original_rels,
                                                 self.num_semantic_rels,
                                                 rel_embedding_dim)

        self.nodes = self.kg_g.ndata['nodes'].to(device)
        self.kg = HGNN(g=self.kg_g,
                       num_edge_types=self.num_semantic_rels,
                       nodes=self.nodes,
                       num_hidden=args.embedding_num,
                       num_layer=num_layer)
        self.kg_size = self.kg.get_output_size()
        self.kg_fc = nn.Sequential(nn.Linear(self.kg_size, self.kg_size),
                                   nn.BatchNorm1d(self.kg_size),
                                   nn.Dropout(dropout),
                                   nn.ReLU())
        self.drug_num = 1253

    def forward(self, datas):
        idx = datas[0]

        original_edge_types = self.kg_g.edata['edges']
        new_edge_types = self.smoother(original_edge_types)

        kg_emb_full = self.kg(new_edge_types)
        kg_emb = kg_emb_full[:self.drug_num]

        smooth_loss=0

        return kg_emb,smooth_loss, idx
class GNN2(nn.Module):
    def __init__(self, smiles, args):
        super(GNN2, self).__init__()

        dropout = 0.1

        self.drug_num = 1253
        num_layer = 3
        self.mol = Mol(smiles, args.embedding_num, num_layer, device, condition='s3')
        self.mol_size = self.mol.gnn.get_output_size()
        self.mol_fc = nn.Sequential(nn.Linear(self.mol_size, self.mol_size),
                                    nn.BatchNorm1d(self.mol_size),
                                    nn.Dropout(dropout),
                                    nn.ReLU(),

                                    nn.Linear(self.mol_size, self.mol_size),
                                    nn.BatchNorm1d(self.mol_size),
                                    nn.Dropout(dropout),
                                    nn.ReLU(),

                                    nn.Linear(self.mol_size, self.mol_size),
                                    nn.BatchNorm1d(self.mol_size),
                                    nn.Dropout(dropout),
                                    nn.ReLU()
                                    )


    def forward(self, arguments):
        kg_emb, smooth_loss, idx= arguments

        mol_emb = self.mol()

        return mol_emb, kg_emb, smooth_loss, idx

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, features, latent_vector):
        x = torch.cat([features, latent_vector], dim=-1)  # Combine features and latent vector
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class FusionLayer(nn.Module):
    def __init__(self, args,bio_feature_tensor):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(args.embedding_num * 6, args.embedding_num * 4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 4, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, 100)).to(device)
        self.bio_feature_tensor = bio_feature_tensor

        self.bio_featureAutoencoder = EmbeddingAutoencoder(self.bio_feature_tensor.shape[1], args.embedding_num)
        self.molAutoencoder = EmbeddingAutoencoder(128, args.embedding_num)

        self.bio_featureAutoencoder.eval()
        self.molAutoencoder.eval()

        self.mutual_info_loss = MutualInformationLoss(args.embedding_num)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        mol_embedding, KG_embedding, smooth_loss, idx = arguments
        self.bio_feature_tensor = self.bio_feature_tensor.to(device)

        self.mol_inputs = mol_embedding.to(device)

        self.bio_featureAutoencoder = self.bio_featureAutoencoder.to(device)
        self.molAutoencoder = self.molAutoencoder.to(device)

        with torch.no_grad():
            bio_feature_embeddings, bio_feature_decoded = self.bio_featureAutoencoder(self.bio_feature_tensor)
            mol_embeddings, mol_decoded = self.molAutoencoder(self.mol_inputs)

            # 计算 reconstruction loss
            loss_bio_feature = F.mse_loss(bio_feature_decoded, self.bio_feature_tensor)
            loss_mol = F.mse_loss(mol_decoded, self.mol_inputs)

            AE_loss = (loss_bio_feature + loss_mol)/2

        bio_feature = bio_feature_embeddings
        mol=mol_embeddings

        loss_mi, z_kg, z_mol,z_bio = self.mutual_info_loss(KG_embedding, mol, bio_feature)

        idx = idx.cpu().numpy().tolist()
        drugA = []
        drugB = []
        for i in idx:
            drugA.append(i[0])
            drugB.append(i[1])

        Embedding = torch.cat(
            [KG_embedding[drugA], bio_feature[drugA], mol[drugA],
             KG_embedding[drugB], bio_feature[drugB], mol[drugB]],
            1).float()

        return self.fullConnectionLayer(Embedding), loss_mi,smooth_loss, AE_loss

class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(EmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class MutualInformationLoss(nn.Module):
    def __init__(self, emb_dim):
        super(MutualInformationLoss, self).__init__()

        self.fc_kg = nn.Linear(emb_dim, emb_dim)
        self.fc_mol = nn.Linear(emb_dim, emb_dim)
        self.fc_bio = nn.Linear(emb_dim, emb_dim)

        self.tanh = nn.Tanh()

    def _pair_mi_loss(self, z_anchor, z_other):
        z_anchor = self.tanh(z_anchor)
        z_other = self.tanh(z_other)

        bi_di_kld = (
            F.kl_div(F.log_softmax(z_anchor, dim=1), F.softmax(z_other, dim=1), reduction='batchmean') +
            F.kl_div(F.log_softmax(z_other, dim=1), F.softmax(z_anchor, dim=1), reduction='batchmean')
        )

        ce_anchor_other = F.mse_loss(z_anchor, z_other.detach())
        ce_other_anchor = F.mse_loss(z_other, z_anchor.detach())

        return ce_anchor_other + ce_other_anchor - bi_di_kld

    def forward(self, z_kg, z_mol, z_bio):
        z_kg = self.fc_kg(z_kg)
        z_mol = self.fc_mol(z_mol)
        z_bio = self.fc_bio(z_bio)

        loss_mol_kg = self._pair_mi_loss(z_mol, z_kg)
        loss_mol_bio = self._pair_mi_loss(z_mol, z_bio)

        total_loss = (loss_mol_kg + loss_mol_bio) / 2

        return total_loss.mean(), z_kg, z_mol, z_bio
