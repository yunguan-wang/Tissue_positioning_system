from tps.find_zones import *
from tps.plotting import *
from tps.segmentation import *
from skimage import io, filters, measure

def pool_masks(l_cv_masks, l_pv_masks):
    # merge two non-overlapping lebeled masks together.
    # ensure there is no overlap
    assert(set(np.unique(l_cv_masks))&set(np.unique(l_pv_masks)) == {0})
    vessel_size_cutoff = l_cv_masks.shape[0] * l_cv_masks.shape[1] / 10000 
    masks = measure.label((l_cv_masks>0) | (l_pv_masks>0))
    output_masks = np.zeros(masks.shape, int)
    cv_labels = []
    pv_labels = []
    for label, region in enumerate(measure.regionprops(masks)):
        if region.label == 0:
            continue
        x0,y0,x1,y1 = region.bbox
        _image = region.image
        if region.area <= vessel_size_cutoff:
            continue
        cv_score = (l_cv_masks[x0:x1,y0:y1][_image]>0).sum()
        pv_score = (l_pv_masks[x0:x1,y0:y1][_image]>0).sum()
        if cv_score > pv_score:
            cv_labels.append(label+1)
        elif cv_score < pv_score:
            pv_labels.append(label+1)
        else:
            continue
        output_masks[x0:x1,y0:y1][_image] = label+1
    cv_masks = output_masks*np.isin(output_masks, cv_labels)
    pv_masks = output_masks*np.isin(output_masks, pv_labels)
    return output_masks, cv_labels, pv_labels, cv_masks, pv_masks

def refine_lobules(cv_masks, min_cv_dist = 50):
    # modify lobules found by watershed by two criteria
    # 1. lobule cv should be distant to each other
    # 2. lobules must be intact, i.e., they cannot be on the image border.

    cv_labels = np.unique(cv_masks)[1:]
    cv_neighbors = [
        ndi.distance_transform_edt(cv_masks != x) for x in cv_labels]

    # nested list comprehension to get a list of list of cv that are too close to each other.
    cv_neighbors = [
        [x] + [y for y in np.unique(cv_masks[cv_neighbors[i] <= min_cv_dist]) if y not in [0,x]
            ] for i,x in enumerate(cv_labels)]
    cv_neighbors = [tuple(sorted(x)) for x in cv_neighbors if len(x)>1]
    cv_neighbors = list(set(cv_neighbors))

    lobules, _, _ = find_lobules(cv_masks)
    xmax, ymax = cv_masks.shape
    for lobule in np.unique(lobules):
        _lobule_mask = (lobules == lobule)
        if _lobule_mask[[0,xmax-1],:].sum() + _lobule_mask[:,[0,ymax-1]].sum() > 0:
            lobules[_lobule_mask] = 0
            continue
        for cn in cv_neighbors:
            if lobule != cn[0]:
                continue
            else:
                lobules[np.isin(lobules, cn)] = cn[0]
    return lobules

def tps_worker(
    img, vessel_size_factor = 1, max_dist=20,dapi_cutoff=20, dapi_dilation_r=0,
    gs_low=0.25, gs_high=0.75, gs_step=0.05, num_zones=9):

    print("Segmentating using GS and DAPI")
    masks, gs_ica, vessels = segmenting_vessels_gs_assisted(
        img,
        vessel_size_t=vessel_size_factor,
        max_dist=max_dist,
        dark_t=dapi_cutoff,
        dapi_dilation_r=dapi_dilation_r,
    )
        # get CV PV classification
    cv_features = extract_features(
        masks, gs_ica, q1=gs_low, q2=gs_high, step=gs_step
    )
    cv_labels, pv_labels = pv_classifier(cv_features.loc[:, "I0":], masks)
    cv_masks = masks * np.isin(masks, cv_labels).copy()
    pv_masks = masks * np.isin(masks, pv_labels).copy()
    # modify CV masks to shrink their borders
    gs_bool, _, _ = extract_gs_channel(img)
    gs_vessel = segmenting_vessels(
        (gs_bool + 0) + (masks == 0 + 0),
        dark_t=1,
        vessel_size_t=1,
        dapi_dilation_r=0,
    )
    masks = shrink_cv_masks(cv_masks, pv_masks, gs_vessel | vessels).astype(int)
    cv_masks = masks * np.isin(masks, cv_labels).copy()
    pv_masks = masks * np.isin(masks, pv_labels).copy()

    zone_crit = calculate_zone_crit(cv_masks, pv_masks, tolerance=100)
    # Calculate zones
    zones = create_zones(
        masks,
        zone_crit,
        cv_labels,
        pv_labels,
        zone_break_type = 'equal_quantile',
        num_zones=num_zones,
    )
    return zones, cv_masks, pv_masks