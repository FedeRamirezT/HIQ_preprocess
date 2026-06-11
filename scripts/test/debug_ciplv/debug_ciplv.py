import numpy
from scipy.signal import hilbert

from init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object
from sEEGnal.tools.mne_tools import prepare_eeg


def compute_ciplv_debug_original(data=None, average_epochs=True, dtype=numpy.float32):
    """
    Original ciPLV computation with debug prints.
    """

    nepochs, nsources, nsamples = data.shape

    triu_idx = numpy.triu_indices(nsources, k=1)
    n_connections = len(triu_idx[0])

    ciplv_sum = numpy.zeros(n_connections, dtype=dtype)

    print('\nRunning ORIGINAL ciPLV\n')

    for iepoch in range(nepochs):

        sourcedata = data[iepoch, :, :]

        if numpy.any(~numpy.isfinite(sourcedata)):
            bad_pos = numpy.argwhere(~numpy.isfinite(sourcedata))
            print(f'[ORIGINAL] NaN/Inf in input data at epoch {iepoch}')
            print(f'First positions [channel, sample]: {bad_pos[:10]}')

        analytic_signal = hilbert(sourcedata, axis=-1)

        amp = numpy.abs(analytic_signal)

        zero_amp = amp == 0
        if numpy.any(zero_amp):
            bad_pos = numpy.argwhere(zero_amp)
            print(f'[ORIGINAL] Zero analytic amplitude at epoch {iepoch}')
            print(f'First positions [channel, sample]: {bad_pos[:10]}')

        sourcevector = analytic_signal / amp

        if numpy.any(~numpy.isfinite(sourcevector)):
            bad_pos = numpy.argwhere(~numpy.isfinite(sourcevector))
            print(f'[ORIGINAL] NaN/Inf in sourcevector at epoch {iepoch}')
            print(f'First positions [channel, sample]: {bad_pos[:10]}')

        cplv = (
            sourcevector @ numpy.conj(sourcevector.T)
        ) / nsamples

        real_plv = numpy.real(cplv)
        imag_plv = numpy.imag(cplv)

        sqrt_argument = 1 - real_plv ** 2

        bad_sqrt = sqrt_argument < 0
        if numpy.any(bad_sqrt):
            bad_pos = numpy.argwhere(bad_sqrt)
            print(f'[ORIGINAL] Negative sqrt argument at epoch {iepoch}')
            print(f'First positions [node_i, node_j]: {bad_pos[:10]}')
            print(f'Example values: {sqrt_argument[bad_sqrt][:10]}')
            print(f'Real PLV values: {real_plv[bad_sqrt][:10]}')

        denom = numpy.sqrt(sqrt_argument)
        denom[denom == 0] = numpy.finfo(dtype).eps

        ciplv_epoch = numpy.abs(imag_plv / denom)

        if numpy.any(~numpy.isfinite(ciplv_epoch)):
            bad_pos = numpy.argwhere(~numpy.isfinite(ciplv_epoch))
            print(f'[ORIGINAL] NaN/Inf in ciPLV matrix at epoch {iepoch}')
            print(f'First positions [node_i, node_j]: {bad_pos[:10]}')

            bad_links = numpy.where(~numpy.isfinite(ciplv_epoch[triu_idx]))[0]
            if len(bad_links) > 0:
                print(f'[ORIGINAL] Bad upper-triangle links at epoch {iepoch}: {bad_links[:10]}')
                for link in bad_links[:10]:
                    i_node = triu_idx[0][link]
                    j_node = triu_idx[1][link]
                    print(
                        f'  link {link}: nodes {i_node}-{j_node}, '
                        f'real={real_plv[i_node, j_node]}, '
                        f'imag={imag_plv[i_node, j_node]}, '
                        f'denom={denom[i_node, j_node]}, '
                        f'ciPLV={ciplv_epoch[i_node, j_node]}'
                    )

        ciplv_sum += ciplv_epoch[triu_idx].astype(dtype)

    ciplv_vector = ciplv_sum / nepochs

    bad_final = numpy.where(~numpy.isfinite(ciplv_vector))[0]

    print('\nORIGINAL result')
    print(f'Number of NaN/Inf links: {len(bad_final)}')
    print(f'Bad final links: {bad_final}')

    return ciplv_vector


def compute_ciplv_debug_fixed(data=None, average_epochs=True, dtype=numpy.float32):
    """
    Fixed ciPLV computation with debug prints.
    """

    nepochs, nsources, nsamples = data.shape

    triu_idx = numpy.triu_indices(nsources, k=1)
    n_connections = len(triu_idx[0])

    ciplv_sum = numpy.zeros(n_connections, dtype=dtype)

    eps = numpy.finfo(dtype).eps

    print('\nRunning FIXED ciPLV\n')

    for iepoch in range(nepochs):

        sourcedata = data[iepoch, :, :]

        if numpy.any(~numpy.isfinite(sourcedata)):
            bad_pos = numpy.argwhere(~numpy.isfinite(sourcedata))
            print(f'[FIXED] NaN/Inf in input data at epoch {iepoch}')
            print(f'First positions [channel, sample]: {bad_pos[:10]}')

        analytic_signal = hilbert(sourcedata, axis=-1)

        amp = numpy.abs(analytic_signal)

        zero_amp = amp == 0
        if numpy.any(zero_amp):
            bad_pos = numpy.argwhere(zero_amp)
            print(f'[FIXED] Zero analytic amplitude corrected at epoch {iepoch}')
            print(f'First positions [channel, sample]: {bad_pos[:10]}')

        amp[zero_amp] = eps

        sourcevector = analytic_signal / amp

        if numpy.any(~numpy.isfinite(sourcevector)):
            bad_pos = numpy.argwhere(~numpy.isfinite(sourcevector))
            print(f'[FIXED] NaN/Inf still present in sourcevector at epoch {iepoch}')
            print(f'First positions [channel, sample]: {bad_pos[:10]}')

        cplv = (
            sourcevector @ numpy.conj(sourcevector.T)
        ) / nsamples

        real_plv = numpy.real(cplv)
        imag_plv = numpy.imag(cplv)

        real_plv_before_clip = real_plv.copy()

        real_plv = numpy.clip(
            real_plv,
            -1 + eps,
            1 - eps
        )

        clipped = real_plv != real_plv_before_clip
        if numpy.any(clipped):
            bad_pos = numpy.argwhere(clipped)
            print(f'[FIXED] Real PLV clipped at epoch {iepoch}')
            print(f'First positions [node_i, node_j]: {bad_pos[:10]}')
            print(f'Original values: {real_plv_before_clip[clipped][:10]}')
            print(f'Clipped values: {real_plv[clipped][:10]}')

        sqrt_argument = 1 - real_plv ** 2
        denom = numpy.sqrt(sqrt_argument)

        denom[denom == 0] = eps

        ciplv_epoch = numpy.abs(imag_plv / denom)

        if numpy.any(~numpy.isfinite(ciplv_epoch)):
            bad_pos = numpy.argwhere(~numpy.isfinite(ciplv_epoch))
            print(f'[FIXED] NaN/Inf in ciPLV matrix at epoch {iepoch}')
            print(f'First positions [node_i, node_j]: {bad_pos[:10]}')

        ciplv_sum += ciplv_epoch[triu_idx].astype(dtype)

    ciplv_vector = ciplv_sum / nepochs

    bad_final = numpy.where(~numpy.isfinite(ciplv_vector))[0]

    print('\nFIXED result')
    print(f'Number of NaN/Inf links: {len(bad_final)}')
    print(f'Bad final links: {bad_final}')

    return ciplv_vector


######################################################################
# Main debug
######################################################################

config, files, sub, ses, task = init()
config['subsystem'] = 'feature_extraction'

# Aquí pon el sujeto problemático
current_sub = '023'
current_ses = '0'
current_task = '1EC'

band_to_debug = 'delta'

BIDS = build_BIDS_object(
    config,
    current_sub,
    current_ses,
    current_task
)

sobi = {
    'desc': 'sobi',
    'components_to_include': ['brain', 'other'],
    'components_to_exclude': []
}

freq_limits_components = [
    config['component_estimation']['low_freq'],
    config['component_estimation']['high_freq']
]

freq_limits_signal = [
    config['feature_extraction']['ciplv']['freq_limits'][0],
    config['feature_extraction']['ciplv']['freq_limits'][-1]
]

raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    channels_to_include=config['global']["channels_to_include"],
    channels_to_exclude=config['global']["channels_to_exclude"],
    freq_limits=freq_limits_components,
    notch_filter=True,
    resample_frequency=config['component_estimation']['resample_frequency'],
    set_annotations=True,
    crop_seconds=config['component_estimation']['crop_seconds'],
    rereference='average'
)

raw = prepare_eeg(
    config,
    BIDS,
    raw=raw,
    apply_sobi=sobi,
    freq_limits=freq_limits_signal,
    metadata_badchannels=True,
    epoch_definition=config['feature_extraction']['ciplv']['epoch_definition']
)

params = config['feature_extraction']['ciplv']['sensor']

iband = params['freq_bands_name'].index(band_to_debug)

banddata = raw.copy().load_data()
banddata.filter(
    params['freq_bands_limits'][iband][0],
    params['freq_bands_limits'][iband][1]
)

data = banddata.get_data()

print('\nData shape:')
print(data.shape)

print('\nInput data finite check:')
print(f'NaN count: {numpy.isnan(data).sum()}')
print(f'Inf count: {numpy.isinf(data).sum()}')

ciplv_original = compute_ciplv_debug_original(
    data=data,
    average_epochs=True
)

ciplv_fixed = compute_ciplv_debug_fixed(
    data=data,
    average_epochs=True
)

print('\nFinal comparison')
print(f'Original NaN/Inf count: {numpy.sum(~numpy.isfinite(ciplv_original))}')
print(f'Fixed NaN/Inf count: {numpy.sum(~numpy.isfinite(ciplv_fixed))}')

bad_original = numpy.where(~numpy.isfinite(ciplv_original))[0]
bad_fixed = numpy.where(~numpy.isfinite(ciplv_fixed))[0]

print(f'Original bad links: {bad_original}')
print(f'Fixed bad links: {bad_fixed}')