import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'#'lasot_extension_subset'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
#"""ostrack""" 160,180,190,200,210,220,230,240,250,260,270,280,290,295,296,297,298,299,300
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-160', run_ids=None, display_name='OSTrack256_160'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-180', run_ids=None, display_name='OSTrack256_180'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-190', run_ids=None, display_name='OSTrack256_190'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-200', run_ids=None, display_name='OSTrack256_200'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-210', run_ids=None, display_name='OSTrack256_210'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-220', run_ids=None, display_name='OSTrack256_220'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-230', run_ids=None, display_name='OSTrack256_230'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-240', run_ids=None, display_name='OSTrack256_240'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-250', run_ids=None, display_name='OSTrack256_250'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-260', run_ids=None, display_name='OSTrack256_260'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-270', run_ids=None, display_name='OSTrack256_270'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-280', run_ids=None, display_name='OSTrack256_280'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-290', run_ids=None, display_name='OSTrack256_290'))
#trackers.extend(trackerlist(name='ostrack', parameter_name='multi-hivit-ep150-4frames-256-6dec-300', dataset_name=dataset_name,
#                            save_name='ostrack_hivitb_v1-295', run_ids=None, display_name='OSTrack256_295'))
trackers.extend(trackerlist(name='aqatrack', parameter_name='multi-hivit-ep150-4frames', dataset_name=dataset_name,
                            save_name='aqatrack_hivitb_v1-145', run_ids=None, display_name='AQATrack256_145'))
trackers.extend(trackerlist(name='aqatrack', parameter_name='multi-hivit-ep150-4frames', dataset_name=dataset_name,
                            save_name='aqatrack_hivitb_v1-146', run_ids=None, display_name='AQATrack256_146'))
trackers.extend(trackerlist(name='aqatrack', parameter_name='multi-hivit-ep150-4frames', dataset_name=dataset_name,
                            save_name='aqatrack_hivitb_v1-147', run_ids=None, display_name='AQATrack256_147'))
trackers.extend(trackerlist(name='aqatrack', parameter_name='multi-hivit-ep150-4frames', dataset_name=dataset_name,
                            save_name='aqatrack_hivitb_v1-148', run_ids=None, display_name='AQATrack256_148'))
dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
