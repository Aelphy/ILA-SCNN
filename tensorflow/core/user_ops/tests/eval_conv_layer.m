clear;
clc;
close all;

% data = importdata('eval_time_conv.txt');
% 
% res = data(:,1)';
% dc = data(:,2)';
% sc = data(:,3)';
% asc = data(:,4)';
% 
% dbpf = data(:,5)';
% dbpi = data(:,6)';
% sbpf = data(:,7)';
% sbpi = data(:,8)';
% sbpc = data(:,9)';
% 
% 
% hold on;
% 
% hTitle  = title ('Runtime on CPU with varying resolution');
% hXLabel = xlabel('resolution : density = 1 / resolution', 'FontSize',13);
% hYLabel = ylabel('Time [s]', 'FontSize',13);
% plot(res, dc, 'c','LineWidth',2);
% plot(res, sc, 'b','LineWidth',2);
% %hLegend = legend('dense conv', 'sparse conv')
% plot(res(:,1:33), dbpf(:, 1:33), 'g','LineWidth',2);
% plot(res, sbpf, 'r','LineWidth',2);
% plot(res(:,1:33), dbpi(:,1:33), 'k','LineWidth',2);
% plot(res, sbpi, 'm','LineWidth',2);
% plot(res, sbpc,  'color', [0.2 0.2 0.6],'LineWidth',2);
% plot(res, asc,'color', [0.5 0.5 0.5], 'LineWidth',2);
% 
% hLegend = legend('dense conv', 'sparse conv', 'dense bp filter', 'sparse bp filter', 'dense bp input', 'sparse bp input','direct sparse backprop','approx conv', 'Location', 'northwest')

data = importdata('eval_time_relu.txt');

res = data(:,1)';
dc = data(:,2)';
sc = data(:,3)';



hold on;

hTitle  = title ('ReLU on CPU with varying resolution');
hXLabel = xlabel('resolution : density = 1 / resolution', 'FontSize',13);
hYLabel = ylabel('Time [s]', 'FontSize',13);
plot(res, dc, 'c','LineWidth',2);
plot(res, sc, 'b','LineWidth',2);

hLegend = legend('dense relu', 'sparse relu', 'Location', 'northwest')


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
