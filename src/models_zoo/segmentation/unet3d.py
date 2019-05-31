import torch.nn as nn
import torch
from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F

class Modified3DUNet(nn.Module):
	def __init__(self, in_channels=1, n_classes=1, base_n_filter = 8):
		super(Modified3DUNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

		# Level 3 context pathway
		self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

		# Level 4 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
		self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
		self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
		self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
		self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
		self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)




	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def forward(self, x):
		#  Level 1 context pathway
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm3d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv3d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm3d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv3d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

		out = self.conv3d_l0(out)
		out = self.inorm3d_l0(out)
		out = self.lrelu(out)

		# Level 1 localization pathway
		out = torch.cat([out, context_4], dim=1)
		out = self.conv_norm_lrelu_l1(out)
		out = self.conv3d_l1(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

		# Level 2 localization pathway
		out = torch.cat([out, context_3], dim=1)
		out = self.conv_norm_lrelu_l2(out)
		ds2 = out
		out = self.conv3d_l2(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

		# Level 3 localization pathway
		out = torch.cat([out, context_2], dim=1)
		out = self.conv_norm_lrelu_l3(out)
		ds3 = out
		out = self.conv3d_l3(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

		#dropout
	#	out = torch.nn.Dropout3d(0.125)(out)

		# Level 4 localization pathway
		out = torch.cat([out, context_1], dim=1)
		out = self.conv_norm_lrelu_l4(out)
		out_pred = self.conv3d_l4(out)

		ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
		ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
		ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
		ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

		out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		seg_layer = out
		out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
		#out = out.view(-1, self.n_classes)
		# out = self.softmax(out)
		return out #, seg_layer


################################################
class SpatialAttentionModified3DUNet(nn.Module):
	def __init__(self, in_channels=1, n_classes=1, base_n_filter=8):
		super(Modified3DUNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
									 bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
									 bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
		self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

		# Level 3 context pathway
		self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2,
								   padding=1, bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
		self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

		# Level 4 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2,
								   padding=1, bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2,
								   padding=1, bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(
			self.base_n_filter * 16, self.base_n_filter * 8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
								   padding=0, bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
		self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1,
								   padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
																							 self.base_n_filter * 4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
		self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1,
								   padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
																							 self.base_n_filter * 2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
		self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1,
								   padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
																							 self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
		self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
								   bias=False)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
										bias=False)
		self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
										bias=False)

		# Attention blocks
		###################
		# Level 1 localization pathway
		self.W_1 = nn.Sequential(
			nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
					  bias=False),
			nn.BatchNorm3d(self.base_n_filter * 8)
		)

		self.psi_1 = nn.Sequential(
			nn.Conv3d(self.base_n_filter * 8, 1, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(1),
			nn.Sigmoid())

		# Level 2 localization pathway
		self.W_2 = nn.Sequential(
			nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
					  bias=False),
			nn.BatchNorm3d(self.base_n_filter * 4)
		)

		self.psi_2 = nn.Sequential(
			nn.Conv3d(self.base_n_filter * 4, 1, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(1),
			nn.Sigmoid())

		# Level 3 localization pathway
		self.W_3 = nn.Sequential(
			nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
					  bias=False),
			nn.BatchNorm3d(self.base_n_filter * 2)
		)

		self.psi_3 = nn.Sequential(
			nn.Conv3d(self.base_n_filter * 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(1),
			nn.Sigmoid())

		# Level 4 localization pathway
		self.W_4 = nn.Sequential(
			nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm3d(self.base_n_filter)
		)

		self.psi_4 = nn.Sequential(
			nn.Conv3d(self.base_n_filter, 1, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(1),
			nn.Sigmoid())

	###################

	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	###################
	def spatial_atten_1(self, out, context):
		out_new = self.W_1(out)
		context_new = self.W_1(context)
		psi = self.lrelu(out_new + context_new)
		psi = self.psi_1(psi)
		context_1 = psi * out
		return context_1

	###################
	def spatial_atten_2(self, out, context):
		out_new = self.W_2(out)
		context_new = self.W_2(context)
		psi = self.lrelu(out_new + context_new)
		psi = self.psi_2(psi)
		context_2 = psi * out
		return context_2

	###################
	def spatial_atten_3(self, out, context):
		out_new = self.W_3(out)
		context_new = self.W_3(context)
		psi = self.lrelu(out_new + context_new)
		psi = self.psi_3(psi)
		context_3 = psi * out
		return context_3

	###################
	def spatial_atten_4(self, out, context):
		out_new = self.W_4(out)
		context_new = self.W_4(context)
		psi = self.lrelu(out_new + context_new)
		psi = self.psi_4(psi)
		context_4 = psi * out
		return context_4

	###################

	def forward(self, x):
		# 		print(x.shape)
		#  Level 1 context pathway
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm3d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv3d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm3d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv3d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

		out = self.conv3d_l0(out)
		out = self.inorm3d_l0(out)
		out = self.lrelu(out)
		#print(out.shape)

		###################
		context_4 = self.spatial_atten_1(out, context_4)
		###################

		# Level 1 localization pathway
		out = torch.cat([out, context_4], dim=1)
		out = self.conv_norm_lrelu_l1(out)
		out = self.conv3d_l1(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

		###################
		context_3 = self.spatial_atten_2(out, context_3)
		###################

		# Level 2 localization pathway
		out = torch.cat([out, context_3], dim=1)
		out = self.conv_norm_lrelu_l2(out)
		ds2 = out
		out = self.conv3d_l2(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

		###################
		context_2 = self.spatial_atten_3(out, context_2)
		###################

		# Level 3 localization pathway
		out = torch.cat([out, context_2], dim=1)
		out = self.conv_norm_lrelu_l3(out)
		ds3 = out
		out = self.conv3d_l3(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

		# dropout
		#	out = torch.nn.Dropout3d(0.125)(out)

		###################
		context_1 = self.spatial_atten_4(out, context_1)
		###################

		# Level 4 localization pathway
		out = torch.cat([out, context_1], dim=1)
		out = self.conv_norm_lrelu_l4(out)
		out_pred = self.conv3d_l4(out)

		ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
		ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
		ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
		ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

		out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		seg_layer = out
		out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)

		# out = out.view(-1, self.n_classes)
		# out = self.softmax(out)
		return out  # , seg_layer


##################################################
class ChannelAttentionModified3DUNet(nn.Module):
	def __init__(self, in_channels=1, n_classes=1, base_n_filter=8, reduction=8):
		super(ChannelAttentionModified3DUNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter
		self.avg3d = nn.AdaptiveAvgPool3d(1)

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
									 bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
									 bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
		self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

		# Level 3 context pathway
		self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
		self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

		# Level 4 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
																							 self.base_n_filter * 8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
		self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
																							 self.base_n_filter * 4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
		self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
																							 self.base_n_filter * 2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
		self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
																							 self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
		self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
								   bias=False)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
										bias=False)
		self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
										bias=False)

		# Attention blocks
		###################
		# Level 1 localization pathway
		self.cSE_1 = nn.Sequential(
			nn.Linear(self.base_n_filter * 8, (self.base_n_filter * 8) // reduction),
			nn.ReLU(inplace=True),
			nn.Linear((self.base_n_filter * 8) // reduction, self.base_n_filter * 8),
			nn.Sigmoid()
		)

		# Level 2 localization pathway
		self.cSE_2 = nn.Sequential(
			nn.Linear(self.base_n_filter * 4, (self.base_n_filter * 4) // reduction),
			nn.ReLU(inplace=True),
			nn.Linear((self.base_n_filter * 4) // reduction, self.base_n_filter * 4),
			nn.Sigmoid()
		)

		# Level 3 localization pathway
		self.cSE_3 = nn.Sequential(
			nn.Linear(self.base_n_filter * 2, (self.base_n_filter * 2) // reduction),
			nn.ReLU(inplace=True),
			nn.Linear((self.base_n_filter * 2) // reduction, self.base_n_filter * 2),
			nn.Sigmoid()
		)

		# Level 4 localization pathway
		self.cSE_4 = nn.Sequential(
			nn.Linear(self.base_n_filter * 1, (self.base_n_filter * 1) // reduction),
			nn.ReLU(inplace=True),
			nn.Linear((self.base_n_filter * 1) // reduction, self.base_n_filter * 1),
			nn.Sigmoid()
		)

	###################

	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	###################
	def channel_atten_1(self, out, context):
		out_new = self.avg3d(out).view(out.shape[0], out.shape[1])
		out_new = self.cSE_1(out_new).view(out.shape[0], out.shape[1], 1, 1, 1)
		# 		print(out_new.shape)

		context_new = self.avg3d(context).view(context.shape[0], context.shape[1])
		context_new = self.cSE_1(context_new).view(context.shape[0], context.shape[1], 1, 1, 1)
		# 		print(context_new.shape)

		psi = self.lrelu(context_new + out_new)
		# 		print(psi.shape)
		psi_new = self.avg3d(psi).view(psi.shape[0], psi.shape[1])
		psi_new = self.cSE_1(psi_new).view(psi.shape[0], psi.shape[1], 1, 1, 1)

		context_1 = psi * out
		return context_1

	###################
	def channel_atten_2(self, out, context):
		out_new = self.avg3d(out).view(out.shape[0], out.shape[1])
		out_new = self.cSE_2(out_new).view(out.shape[0], out.shape[1], 1, 1, 1)

		context_new = self.avg3d(context).view(context.shape[0], context.shape[1])
		context_new = self.cSE_2(context_new).view(context.shape[0], context.shape[1], 1, 1, 1)

		psi = self.lrelu(context_new + out_new)

		psi_new = self.avg3d(psi).view(psi.shape[0], psi.shape[1])
		psi_new = self.cSE_2(psi_new).view(psi.shape[0], psi.shape[1], 1, 1, 1)

		context_2 = psi * out
		return context_2

	###################
	def channel_atten_3(self, out, context):
		out_new = self.avg3d(out).view(out.shape[0], out.shape[1])
		out_new = self.cSE_3(out_new).view(out.shape[0], out.shape[1], 1, 1, 1)

		context_new = self.avg3d(context).view(context.shape[0], context.shape[1])
		context_new = self.cSE_3(context_new).view(context.shape[0], context.shape[1], 1, 1, 1)

		psi = self.lrelu(context_new + out_new)

		psi_new = self.avg3d(psi).view(psi.shape[0], psi.shape[1])
		psi_new = self.cSE_3(psi_new).view(psi.shape[0], psi.shape[1], 1, 1, 1)

		context_3 = psi * out
		return context_3

	###################
	def channel_atten_4(self, out, context):
		out_new = self.avg3d(out).view(out.shape[0], out.shape[1])
		out_new = self.cSE_4(out_new).view(out.shape[0], out.shape[1], 1, 1, 1)

		context_new = self.avg3d(context).view(context.shape[0], context.shape[1])
		context_new = self.cSE_4(context_new).view(context.shape[0], context.shape[1], 1, 1, 1)

		psi = self.lrelu(context_new + out_new)

		psi_new = self.avg3d(psi).view(psi.shape[0], psi.shape[1])
		psi_new = self.cSE_4(psi_new).view(psi.shape[0], psi.shape[1], 1, 1, 1)

		context_4 = psi * out
		return context_4

	###################

	def forward(self, x):
		#  Level 1 context pathway
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm3d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv3d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm3d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv3d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

		out = self.conv3d_l0(out)
		out = self.inorm3d_l0(out)
		out = self.lrelu(out)

		###################
		context_4 = self.channel_atten_1(out, context_4)
		###################

		# Level 1 localization pathway
		out = torch.cat([out, context_4], dim=1)
		out = self.conv_norm_lrelu_l1(out)
		out = self.conv3d_l1(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

		###################
		context_3 = self.channel_atten_2(out, context_3)
		###################

		# Level 2 localization pathway
		out = torch.cat([out, context_3], dim=1)
		out = self.conv_norm_lrelu_l2(out)
		ds2 = out
		out = self.conv3d_l2(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

		###################
		context_2 = self.channel_atten_3(out, context_2)
		###################

		# Level 3 localization pathway
		out = torch.cat([out, context_2], dim=1)
		out = self.conv_norm_lrelu_l3(out)
		ds3 = out
		out = self.conv3d_l3(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

		# dropout
		#	out = torch.nn.Dropout3d(0.125)(out)

		###################
		context_1 = self.channel_atten_4(out, context_1)
		###################

		# Level 4 localization pathway
		out = torch.cat([out, context_1], dim=1)
		out = self.conv_norm_lrelu_l4(out)
		out_pred = self.conv3d_l4(out)

		ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
		ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
		ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
		ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

		out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		seg_layer = out
		out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)

		# out = out.view(-1, self.n_classes)
		# out = self.softmax(out)
		return out  # , seg_layer


#### FPN

class Modified3DUNet_FPN(nn.Module):
	def __init__(self, in_channels=1, n_classes=1, base_n_filter=8):
		super(Modified3DUNet_FPN, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
									 bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
									 bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
		self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

		# Level 3 context pathway
		self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
		self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

		# Level 4 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
								   bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
																							 self.base_n_filter * 8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

		# FPN
		self.pyramid_t1 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 8, kernel_size=1)
		self.pyramid_t2 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 8, kernel_size=1)
		self.pyramid_t3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=1)
		self.pyramid_t4 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1)
		self.pyramid_t5 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=(1, 1, 1))

		self.s5 = self.segmentation_block(self.base_n_filter * 8, self.base_n_filter * 4, n_upsamples=3)
		self.s4 = self.segmentation_block(self.base_n_filter * 8, self.base_n_filter * 4, n_upsamples=3)
		self.s3 = self.segmentation_block(self.base_n_filter * 8, self.base_n_filter * 4, n_upsamples=1)
		self.s2 = self.segmentation_block(self.base_n_filter * 8, self.base_n_filter * 4, n_upsamples=0)
		self.s1 = self.segmentation_block(self.base_n_filter * 8, self.base_n_filter * 4, n_upsamples=0)

		self.dropout = nn.Dropout3d(p=0.2, inplace=True)
		self.final_conv = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, padding=0)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
		self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
																							 self.base_n_filter * 4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
		self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
																							 self.base_n_filter * 2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
		self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
								   bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
																							 self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
		self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
								   bias=False)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
										bias=False)
		self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
										bias=False)

	def segmentation_block(self, feat_in, feat_out, n_upsamples):
		blocks = [self.conv_norm_lrelu(feat_in, feat_out, upsample=bool(n_upsamples))]
		if n_upsamples > 1:
			for _ in range(1, n_upsamples):
				blocks.append(self.conv_norm_lrelu(feat_out, feat_out, upsample=True))
		return nn.Sequential(*blocks)

	def conv_norm_lrelu(self, feat_in, feat_out, upsample=False):
		block = [nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
				 nn.InstanceNorm3d(feat_out),
				 nn.LeakyReLU()
				 ]
		if upsample:
			block.append(nn.Upsample(scale_factor=2, mode='nearest'))
		return nn.Sequential(*block)

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def forward(self, x):
		#  Level 1 context pathway
		# print(x.shape)
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)
		context_1 = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm3d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv3d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm3d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv3d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		context_5 = out

		# FPN
		p5 = self.pyramid_t5(context_5)
		p5 = F.interpolate(p5, scale_factor=2, mode='nearest')
		p4 = self.pyramid_t4(context_4)
		p4 = p4 + p5
		# p4 = F.interpolate(p4, scale_factor=2, mode='nearest')
		p3 = self.pyramid_t4(context_4)
		p3 = p3 + p4
		p3 = F.interpolate(p3, scale_factor=4, mode='nearest')
		p2 = self.pyramid_t2(context_2)
		p2 = p2 + p3
		p2 = F.interpolate(p2, scale_factor=2, mode='nearest')
		p1 = self.pyramid_t1(context_1)
		p1 = p1 + p2
		# p1 = F.interpolate(p1, scale_factor=2, mode='nearest')

		#print(p1.shape, p2.shape, p3.shape, p4.shape, p5.shape)

		s5 = self.s5(p5)
		# print(p5.shape, s5.shape)
		s4 = self.s4(p4)
		# print(p4.shape, s4.shape)
		s3 = self.s3(p3)
		# print(p3.shape, s3.shape)
		s2 = self.s2(p2)
		# print(p2.shape, s2.shape)
		s1 = self.s2(p1)
		# print(p1.shape, s1.shape)

		out = s5 + s4 + s3 + s2 + s1
		out = self.dropout(out)
		out = self.final_conv(out)
		# out = F.interpolate(out, scale_factor=4, mode='nearest', align_corners=True)
		# print(out.shape)

		# From here I
		# out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
		# out = self.conv3d_l0(out)
		# out = self.inorm3d_l0(out)
		# out = self.lrelu(out)

		# # Level 1 localization pathway
		# out = torch.cat([out, context_4], dim=1)
		# out = self.conv_norm_lrelu_l1(out)
		# out = self.conv3d_l1(out)
		# out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

		# # Level 2 localization pathway
		# out = torch.cat([out, context_3], dim=1)
		# out = self.conv_norm_lrelu_l2(out)
		# ds2 = out
		# out = self.conv3d_l2(out)
		# out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

		# # Level 3 localization pathway
		# out = torch.cat([out, context_2], dim=1)
		# out = self.conv_norm_lrelu_l3(out)
		# ds3 = out
		# out = self.conv3d_l3(out)
		# out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

		# #dropout
		# #	out = torch.nn.Dropout3d(0.125)(out)

		# # Level 4 localization pathway
		# out = torch.cat([out, context_1], dim=1)
		# out = self.conv_norm_lrelu_l4(out)
		# out_pred = self.conv3d_l4(out)

		# ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
		# ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
		# ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
		# ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		# ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

		# out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		# seg_layer = out
		# out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
		# #out = out.view(-1, self.n_classes)
		# # out = self.softmax(out)
		return out  # , seg_layer

# x = Variable(torch.rand(1, 1, 64, 224, 224).cuda())
# model = Modified3DUNet().cuda()
# print(summary(model, (1, 16, 224, 224)))
