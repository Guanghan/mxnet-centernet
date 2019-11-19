import sys
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/")

from models.decoder import *
from models.tensor_utils import _gather_feat
from test_hourglass_network import Y
from models.decoder import _topk, decode_centernet

def test_topk():
    scores = nd.random.uniform(shape=(16,80,128,128))
    batch, cat, height, width = scores.shape
    [topk_scores, topk_inds] = nd.topk(nd.reshape(scores, (batch, cat, -1)), ret_typ='both', k=10)  # return both value and indices
    print("topk_scores.shape = ", topk_scores.shape)
    print("topk_inds.shape = ", topk_inds.shape)

    topk_score, topk_inds, topk_clses, topk_ys, topk_xs = _topk(scores, K=10)
    print("topk_score.shape = ", topk_score.shape)

def test_tensor_utils():
    scores = nd.random.uniform(shape=(16,80,128,128))
    batch, cat, height, width = scores.shape
    [topk_scores, topk_inds] = nd.topk(nd.reshape(scores, (batch, cat, -1)), ret_typ='both', k=10)  # return both value and indices
    print("topk_scores.shape = ", topk_scores.shape)
    print("topk_inds.shape = ", topk_inds.shape)
    #print("max(topk_inds)", max(topk_inds[1,1,:]))
    topk_inds = topk_inds % (height * width)
    print("topk_inds.shape = ", topk_inds.shape)
    #print("max(topk_inds)", max(topk_inds[2,2,:]))
    topk_ys   = (topk_inds / width).astype('int').astype('float32')
    topk_xs   = (topk_inds % width).astype('int').astype('float32')
    print("topk_ys.shape = ", topk_ys.shape)
    print("topk_xs.shape = ", topk_xs.shape)

    [topk_score, topk_ind] = nd.topk(nd.reshape(topk_scores, (batch, -1)), ret_typ='both', k= 10)
    print("topk_ind.shape = ", topk_ind.shape)
    #print("max(topk_ind) = ", max(topk_ind[1,:]))

    #topk_inds = _gather_feat(nd.reshape(topk_inds, (batch, -1, 1)), topk_ind)
    #print("topk_inds.shape = ", topk_inds.shape)

    #data = topk_inds.reshape((-1, 1))
    #index = topk_ind.reshape((-1, 1))
    data = topk_inds.reshape((batch, -1))
    index = topk_ind.reshape((batch, -1))
    print(data.shape)
    print(index.shape)
    topk_inds = index.zeros_like()
    #topk_inds = nd.take(data, index, axis = 0, mode='wrap')
    topk_inds = nd.take(data, index, axis = 1, mode='wrap')
    #topk_inds = topk_inds.reshape((batch, -1, 1))
    print("topk_inds.shape = ", topk_inds.shape)

    aa = nd.split(topk_inds, axis=2, num_outputs=10)
    print(len(aa))
    bb = [nd.linalg.extractdiag(nd.squeeze(item)) for item in aa]
    print(bb[0].shape)
    cc = nd.stack(*bb).expand_dims(2)
    print(cc.shape)

    hm_score = nd.random.uniform(shape=(16,80,128*128))
    min_ind = nd.random.uniform(shape=(16,80,100))

    hm_score = hm_score.reshape((16*80, 128*128))
    min_ind = min_ind.reshape((16*80, 10))
    hm_score_selected = nd.take(hm_score, min_ind, axis=1, mode='wrap')
    print(hm_score_selected.shape)
    print(hm_score_selected.reshape(1280, 10).shape)
    '''
    aa = nd.split(hm_score_selected, axis=2, num_outputs=10)
    print(len(aa))
    bb = [nd.linalg.extractdiag(nd.squeeze(item)) for item in aa]
    cc = nd.stack(*bb).expand_dims(2)
    print(cc.shape)
    '''

    return


def test_decode_centernet():
    print("output: heatmaps", Y[0]["hm"].shape)
    print("output: wh_scale", Y[0]["wh"].shape)
    print("output: xy_offset", Y[0]["reg"].shape)

    heatmaps, scale, offset = Y[0]["hm"], Y[0]["wh"], Y[0]["reg"]
    detections = decode_centernet(heat=heatmaps, wh=scale, reg=offset, cat_spec_wh=False, K=10, flag_split=False)
    print(detections.shape)


def test_decode_centernet_pose():
    center_heatmaps = nd.random.normal(shape=(16, 1, 128, 128))  # 1 class: human
    scale = nd.random.normal(shape=(16, 2, 128, 128))
    offset = nd.random.normal(shape=(16, 2, 128, 128))

    pose_relative = nd.random.normal(shape=(16, 34, 128, 128))
    pose_heatmaps = nd.random.normal(shape = (16, 17, 128, 128))
    pose_offset = nd.random.normal(shape=(16, 2, 128, 128))

    detections = decode_centernet_pose(center_heatmaps, scale, pose_relative, offset, pose_heatmaps, pose_offset, K=100)
    # (bboxes, scores, kps, clses, dim=2)
    print(detections.shape)
    return


def test_inference():
    from opts import opts
    from detectors.pose_detector import PoseDetector
    opt = opts()
    opt = opt.init()
    detector = PoseDetector(opt)
    detector.model = load_model(detector.model, opt.pretrained_path, ctx = ctx)

    img_path = "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/assets/demo.jpg"
    ret = detector.run(img_path)
    results[img_id] = ret['results']

    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    print_message = ''
    for t in avg_time_stats:
        avg_time_stats[t].update(ret[t])
        print_message += '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    print(print_message)
    return


def display_func_name(func):
    print("\n \t testing {}".format(func.__name__))


if __name__ == "__main__":
    #function_list = [test_topk, test_decode_centernet]
    #function_list = [test_decode_centernet]
    #function_list = [test_decode_centernet_pose]
    #function_list = [test_inference]
    #function_list = [test_topk]
    function_list = [test_tensor_utils]
    for unit_test in function_list:
        display_func_name(unit_test)
        unit_test()
