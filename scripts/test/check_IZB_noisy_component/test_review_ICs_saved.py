"""Test the CURRENT scripts/shared/review_ICs.py without refitting ICA.

Place in scripts/test/check_IZB_noisy_component. From that folder:
    python test_review_ICs_saved.py --subjects 023 024
    python test_review_ICs_saved.py                 # all saved final ICAs

Optional --project-root PATH, --config FILE, --output NEW_DIRECTORY.
EEG/ICA inputs and the pipeline's QC files are never modified. The saved
ICA must have been fitted with the current configuration and bad channels.

By default reconstructs the classification BEFORE review_ICs from saved
ICLabel scores and unclear_threshold. This avoids accumulating previous
review decisions. --keep-saved-labels instead starts from the FIF labels.
Every invocation imports the current review_ICs code anew.

Figures use sEEGnal's installed _compute_strict_window_spectrum QC estimator
and its amplitude spectral density (sqrt PSD), grouping ICs by category.
This helper is private: an incompatible future sEEGnal version may require
updating this script. The actual detector uses its own calculation unchanged.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import html
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ['brain','muscle','eog','ecg','line_noise','ch_noise','other']


def reset_classification(ica, threshold):
    if not hasattr(ica,'labels_scores_'):
        raise ValueError('Missing ICLabel scores; cannot reconstruct baseline. '
                         'Restore its TSV or explicitly use --keep-saved-labels.')
    keys=list(ica.labels_)  # read_ica orders score columns to these labels.
    scores=np.asarray(ica.labels_scores_)
    if scores.shape!=(ica.n_components_,len(keys)) or not np.isfinite(scores).all():
        raise ValueError('Invalid ICLabel score dimensions or values')
    labels={key:[] for key in dict.fromkeys(keys+CATEGORIES)}
    for component, probabilities in enumerate(scores):
        index=int(probabilities.argmax())
        label=keys[index] if probabilities[index]>=threshold else 'other'
        labels[label].append(component)
    ica.labels_=labels


def plot_qc(psd, freqs, labels, title, path):
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for label,ax in zip(CATEGORIES,axes.flat):
        indices=labels.get(label,[])
        if indices:
            ax.plot(freqs,np.sqrt(psd[indices]).T,alpha=.2)
        ax.set_title(f'{label} (n={len(indices)})')
        ax.set(xlim=(freqs[0],freqs[-1]),xlabel='Frequency (Hz)',
               ylabel='Amplitude spectral density (AU/√Hz)')
        ax.grid(alpha=.3)
    axes.flat[-1].axis('off')
    fig.suptitle(title,fontsize=12)
    fig.savefig(path,dpi=130)
    plt.close(fig)


def json_default(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    folder=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=folder.parents[2])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--subjects',nargs='+',help='Subject IDs, e.g. 023 024')
    parser.add_argument('--output',type=Path,
                        default=folder/('review_test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--keep-saved-labels',action='store_true')
    args=parser.parse_args()
    root=args.project_root.expanduser().resolve()
    config_file=args.config or root/'scripts/run/init/config.json'
    config=json.loads(config_file.read_text(encoding='utf-8'))
    data_root=Path(config['path']['data_root']).expanduser()
    if not data_root.is_absolute(): data_root=root/'scripts/run'/data_root
    data_root=data_root.resolve()
    config['path'].update(data_root=str(data_root),sourcedata=str(data_root/'sourcedata/eeg'),
                          recordings=str(data_root/'recordings.tsv'))
    config['subsystem']='preprocess'
    sys.path.insert(0,str(root))
    import mne
    from mne_bids import get_bids_path_from_fname
    from sEEGnal.tools import bids_tools,mne_tools,qc_tools
    module_path=root/'scripts/shared/review_ICs.py'
    spec=importlib.util.spec_from_file_location('current_project_review_ICs',module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files=sorted((data_root/'derivatives/sEEGnal/preprocess').rglob('*_desc-cleaning_ica.fif'))
    wanted={s.removeprefix('sub-') for s in args.subjects} if args.subjects else None
    if wanted:
        files=[p for p in files if get_bids_path_from_fname(p,check=False).subject in wanted]
        found={get_bids_path_from_fname(p,check=False).subject for p in files}
        if wanted-found: parser.error('No ICA found for: '+', '.join(sorted(wanted-found)))
    if not files: parser.error('No saved desc-cleaning ICA found')
    args.output.mkdir(parents=True,exist_ok=False)
    # Snapshot the tested detector/settings for reproducible interpretation.
    (args.output/'review_ICs_used.py').write_text(module_path.read_text(encoding='utf-8'),encoding='utf-8')
    (args.output/'config_used.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    records=[]
    for number,path in enumerate(files,1):
        print(f'[{number}/{len(files)}] {path.name}',flush=True)
        raw=None
        record={'ica_path':str(path),'status':'error'}
        try:
            with mne.utils.use_log_level('ERROR'):
                bids=get_bids_path_from_fname(path,check=False)
                bids.update(root=data_root,description=None,suffix='eeg',extension='.vhdr',check=False)
                ica=bids_tools.read_ica(config,bids,desc='cleaning')
                if not args.keep_saved_labels:
                    reset_classification(ica,config['component_estimation']['unclear_threshold'])
                before=deepcopy(ica.labels_)
                ce=config['component_estimation']
                raw=mne_tools.prepare_eeg(
                    config,bids,preload=True,
                    channels_to_include=config['global']['channels_to_include'],
                    channels_to_exclude=config['global']['channels_to_exclude'],
                    resample_frequency=ce['resample_frequency'],notch_filter=True,
                    freq_limits=[ce['low_freq'],ce['high_freq']],crop_seconds=ce['crop_seconds'],
                    metadata_badchannels=True,exclude_badchannels=True,
                    set_annotations=True,rereference='average',
                )
                if raw.ch_names!=ica.ch_names:
                    raise ValueError('Prepared channel names/order differ from the saved ICA')
                # Sources and QC spectra computed once, shared by before/after.
                sources=ica.get_sources(raw)
                psd,freqs=qc_tools._compute_strict_window_spectrum(sources)
                del sources
                ica=module.review_ICs(raw,ica)
                after=deepcopy(ica.labels_)
            stem=path.name.removesuffix('_desc-cleaning_ica.fif')
            before_file=stem+'_before.png';after_file=stem+'_after.png'
            plot_qc(psd,freqs,before,stem+' | Before review_ICs',args.output/before_file)
            plot_qc(psd,freqs,after,stem+' | After review_ICs',args.output/after_file)
            moved=[{'component':i,'before':[k for k,v in before.items() if i in v],
                     'after':[k for k,v in after.items() if i in v]}
                   for i in range(ica.n_components_)
                   if [k for k,v in before.items() if i in v]!=[k for k,v in after.items() if i in v]]
            record.update(status='ok',recording=stem,n_components=ica.n_components_,
                          before=before,after=after,changes=moved,
                          before_image=before_file,after_image=after_file,
                          detector_metrics=getattr(ica,'stable_noise_report_',None))
            (args.output/(stem+'_labels.json')).write_text(json.dumps(record,indent=2,default=json_default),encoding='utf-8')
        except Exception as error:
            record.update(error=str(error),traceback=traceback.format_exc())
            print(f'Error: {error}',flush=True)
        finally:
            if raw is not None: raw.close()
        records.append(record)
        (args.output/'summary.json').write_text(json.dumps(records,indent=2,default=json_default),encoding='utf-8')
    content='<!doctype html><meta charset="utf-8"><title>review_ICs test</title><style>body{font-family:system-ui;margin:24px}img{width:100%;max-width:1800px}</style>'
    content+='<h1>Current review_ICs applied to saved ICA</h1><p>No ICA fitting. Inputs unchanged. '
    content+=('Baseline: saved FIF labels.' if args.keep_saved_labels else 'Baseline: reconstructed from saved ICLabel scores and unclear_threshold.')+'</p>'
    for record in records:
        content+='<h2>'+html.escape(Path(record['ica_path']).name)+'</h2>'
        if record['status']!='ok':
            content+='<p>Error: '+html.escape(record['error'])+'</p>';continue
        newly=set(record['after'].get('line_noise',[]))-set(record['before'].get('line_noise',[]))
        content+=f'<p>line_noise: {len(record["before"].get("line_noise",[]))} → {len(record["after"].get("line_noise",[]))} / {record["n_components"]}. Added ICs (zero-based): {sorted(newly)}</p>'
        for key in ['before_image','after_image']:
            name=html.escape(record[key],quote=True)
            content+=f'<a href="{name}"><img loading="lazy" src="{name}"></a>'
    (args.output/'review.html').write_text(content,encoding='utf-8')
    print('Open: '+str((args.output/'review.html').resolve()))
    return int(any(r['status']!='ok' for r in records))


if __name__=='__main__':
    raise SystemExit(main())
