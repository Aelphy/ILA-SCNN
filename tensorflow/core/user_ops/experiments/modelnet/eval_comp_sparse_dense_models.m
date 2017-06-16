clear;
clc;
close all;

data = importdata('res_training.txt');

epoch = data(:,1)';
loss_s = data(:,2)';
loss_d = data(:,3)';
loss_sa = data(:,4)';
oa_s = data(:,7)';
oa_d = data(:,10)';
oa_sa = data(:,13)';



hold on;

hTitle  = title ('Loss of Models');
hXLabel = xlabel('Epoch', 'FontSize',13);
hYLabel = ylabel('Loss', 'FontSize',13);
plot(epoch, loss_s, 'c','LineWidth',2);
plot(epoch, loss_d, 'b','LineWidth',2);
plot(epoch, loss_sa, 'r','LineWidth',2);

hLegend = legend('exact sparse', 'dense', 'approximated sparse', 'Location', 'northwest')

% hTitle  = title ('Overall Accuracy of Models');
% hXLabel = xlabel('Epoch', 'FontSize',13);
% hYLabel = ylabel('Overall Accuracy', 'FontSize',13);
% plot(epoch, oa_s, 'c','LineWidth',2);
% plot(epoch, oa_d, 'b','LineWidth',2);
% plot(epoch, oa_sa, 'r','LineWidth',2);
% 
% hLegend = legend('exact sparse', 'dense', 'approximated sparse', 'Location', 'southeast')



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
print -depsc2 'eval_time_conv.eps'
