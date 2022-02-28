class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model =  pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        self.regression_layer = nn.Sequential(nn.Linear(512, 9))

    def forward(self, x):
        batch_size ,_,_,_ = x.shape #taking out batch_size from input image
        x = self.model.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1) # then reshaping the batch_size
        x = self.regression_layer(x)
        x_transl = x[:, -3:]
        x_rot = compute_rotation_matrix_from_ortho6d(x[:, :6].view(
            batch_size, -1))

        return x_transl, x_rot

    def compute_rotation_matrix_l2_loss(self, gt_rotation_matrix, predict_rotation_matrix):
        loss_function = nn.MSELoss()
        loss = loss_function(predict_rotation_matrix, gt_rotation_matrix)

        return loss

    def compute_rotation_matrix_geodesic_loss(self, gt_rotation_matrix, predict_rotation_matrix):
        theta = compute_geodesic_distance_from_two_matrices(gt_rotation_matrix, predict_rotation_matrix)
        error = theta.mean()

        return error