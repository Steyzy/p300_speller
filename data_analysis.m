%=========================================================================================================
% this file is for data analysis on training accuracies
%=========================================================================================================

% graph_accs = cell(size(all_accs{1}, 1), 1);
% for i=1:size(all_accs{1}, 1)
%     graph_accs{i} = cell(size(all_accs, 1), 1);
%     for z=1:size(all_accs, 1)
%         if isempty(all_accs{z})
%             continue
%         end
%         graph_accs{i}{z} = all_accs{z}{i};
%     end
% end
% save("/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/results/graph_accs.mat", "graph_accs");


load("/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/results/graph_accs.mat");
for i=1:size(graph_accs, 1)
    %create figure, multiple subjects in one plot
    figure; 
    ylim([0 1]);
    hold on;
    for z=1:size(graph_accs{1}, 1)
        plot(graph_accs{i}{z});   %plot the accuracy over the iterations
    end
    saveas(gcf, sprintf('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/graphs/nLetters_%d.jpg', i));
    close;
end