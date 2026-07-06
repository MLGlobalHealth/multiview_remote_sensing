import numpy as np, pandas as pd, os, glob
FEAT='/data/coml-satellites/shug7409/multiview_remote_sensing/modelling/dino/features'
OUT='/data/coml-hawkes/stat0278/multiview_remote_sensing/modelling/dino/v3_prepped'
master=pd.read_csv('survey_processing/processed_data/dhs_processed.csv')
M=master[['CENTROID_ID','deprived_sev','deprived_sev_k','deprived_sev_n',
          'deprived_mod','deprived_mod_k','deprived_mod_n']]
assert M.CENTROID_ID.is_unique, 'master CENTROID_ID not unique'
dirs=sorted(glob.glob(FEAT+'/spatial_[LS]_fold*_bands*'))  # both Sentinel and Landsat
print('found', len(dirs), 'dirs')
for d in dirs:
    name=os.path.basename(d); o=os.path.join(OUT,name); os.makedirs(o,exist_ok=True)
    if os.path.exists(f'{o}/X_train.csv') and os.path.exists(f'{o}/y_train.csv'):
        print('skip',name,'(exists)'); continue
    for sp in ['train','test']:
        X=np.load(f'{d}/X_{sp}.npy')
        meta=pd.read_csv(f'{d}/meta_{sp}.csv')
        assert len(meta)==X.shape[0], f'{name} {sp} row mismatch {len(meta)} vs {X.shape[0]}'
        y=meta[['CENTROID_ID','YEAR']].merge(M,on='CENTROID_ID',how='left')
        assert len(y)==len(meta), f'{name} {sp} merge expanded rows'
        assert y.deprived_sev_n.notna().all() and y.deprived_mod_n.notna().all(), f'{name} {sp} missing counts'
        pd.DataFrame(X,columns=[str(i) for i in range(X.shape[1])]).to_csv(f'{o}/X_{sp}.csv',index=False)
        y.to_csv(f'{o}/y_{sp}.csv',index=False)
    print('done',name,'dim',X.shape[1])
print('ALL PREPPED')
