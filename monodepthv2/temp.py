# import numpy.ma as ma
#
# mask = mask.asnumpy().astype(np.bool)
# mask = ~mask
#
# depth_gt = depth_gt.asnumpy()
# masked_depth_gt = ma.masked_array(depth_gt, mask)
# depth_gt = masked_depth_gt.compressed()
# depth_gt = mx.nd.array(depth_gt)
# print(depth_gt)
#
# depth_pred = depth_pred.asnumpy()
# masked_depth_pred = ma.masked_array(depth_pred, mask)
# depth_pred = masked_depth_pred.compressed()
# depth_pred = mx.nd.array(depth_pred)
# print(depth_pred)
#
# print(time.time() - tic)
# exit()

# crop_mask = mx.nd.zeros_like(mask)
# crop_mask[:, :, 153:371, 44:1197] = 1
# mask = mask * crop_mask
#
# # depth_gt = depth_gt[mask]
# depth_gt = mx.nd.contrib.boolean_mask(depth_gt.reshape(-1), mask.reshape(-1))
# depth_pred = mx.nd.contrib.boolean_mask(depth_pred.reshape(-1), mask.reshape(-1))
#
# scale_factor = np.median(depth_gt.asnumpy()) / np.median(depth_pred.asnumpy())
# depth_pred *= scale_factor
#
# depth_pred = mx.nd.clip(depth_pred, a_min=1e-3, a_max=80)
#
# depth_errors = compute_depth_errors(depth_gt, depth_pred)
#
# for i, metric in enumerate(self.depth_metric_names):
#     depth_metrics[metric] += np.array(depth_errors[i].cpu())


"""loading weights
encoder_path = os.path.join("./models/mono+stereo_640x192_mx", "encoder.params")
decoder_path = os.path.join("./models/mono+stereo_640x192_mx", "depth.params")
self.models["encoder"].load_parameters(encoder_path, ctx=self.opt.ctx)
self.models["depth"].load_parameters(decoder_path, ctx=self.opt.ctx)
"""