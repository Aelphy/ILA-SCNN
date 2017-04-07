clear;
clc;
close all;

t_p_hash = importdata('build/result_hash_map.txt');
t_s_map = importdata('build/result_map.txt');
t_p_merge_sort = importdata('build/result_merge_sort.txt');
t_sparse_kernel = importdata('build/result_sparse_kernel_merge_sort.txt');



hold on;


% hTitle  = title ('Runtime on CPU');
% hXLabel = xlabel('number of threads', 'FontSize',13);
% hYLabel = ylabel('time [s]', 'FontSize',13);
% 
% num_cores = 1 : length(t_p_hash);
% plot(num_cores, t_p_hash, 'b','LineWidth',2);
% 
% num_cores = 1 : length(t_s_map);
% plot(num_cores, t_s_map, 'xr','LineWidth',2);
% 
% num_cores = 1 : length(t_p_merge_sort);
% plot(num_cores, t_p_merge_sort, 'c','LineWidth',2);
% 
% hLegend = legend('parallel hash map', 'serial ordered map', 'parallel merge sort', 'Location', 'northeast')



hTitle  = title ('Runtime on CPU with varying Filter Density');
hXLabel = xlabel('Sparsity of Filter', 'FontSize',13);
hYLabel = ylabel('Time [s]', 'FontSize',13);

num_samples = 1 : length(t_sparse_kernel);
density_samples = 1 - num_samples / length(t_sparse_kernel);
plot(fliplr(density_samples), fliplr(t_sparse_kernel), 'c','LineWidth',2);


set( gca                       , ...
    'FontName'   , 'Helvetica' );
%set([hTitle, hXLabel, hYLabel], ...
%    'FontName'   , 'AvantGarde');
% set([hLegend, gca]             , ...
%     'FontSize'   , 14          , ...
%     'Box', 'off');
set([hXLabel, hYLabel]  , ...
    'fontweight','Bold'       , ...
    'FontSize'   , 13          );
set( hTitle                    , ...
    'FontSize'   , 16          );

set(gca, ...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'XColor'      , [.3 .3 .3], ...
  'YColor'      , [.3 .3 .3], ...
  'LineWidth'   , 2         );

set(gcf, 'PaperPositionMode', 'auto');
print -depsc2 'build/finalPlot1.eps'