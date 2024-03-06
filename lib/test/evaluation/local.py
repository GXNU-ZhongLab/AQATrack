from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/data/got10k_lmdb'
    settings.got10k_path = '/home/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/data/itb'
    settings.lasot_extension_subset_path = '/home/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/data/lasot_lmdb'
    settings.lasot_path = '/home/data/LaSOTBenchmark'
    settings.network_path = '/home/data/workspace/xjx/code/hivit/aqatrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/data/nfs'
    settings.otb_path = '/home/data/otb'
    settings.prj_dir = '/home/data/workspace/xjx/code/hivit/aqatrack'
    settings.result_plot_path = '/home/data/workspace/xjx/code/hivit/aqatrack/output/test/result_plots'
    settings.results_path = '/home/data/workspace/xjx/code/hivit/aqatrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/data/workspace/xjx/code/hivit/aqatrack/output'
    settings.segmentation_path = '/home/data/workspace/xjx/code/hivit/aqatrack/output/test/segmentation_results'
    settings.tc128_path = '/home/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/data/trackingnet'
    settings.uav_path = '/home/data/uav'
    settings.vot18_path = '/home/data/vot2018'
    settings.vot22_path = '/home/data/vot2022'
    settings.vot_path = '/home/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

