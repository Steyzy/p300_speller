%=========================================================================================================
% this file is for data analysis on training accuracies
%=========================================================================================================

num_subj = size(all_accs, 1);
fLetters = size(all_accs{1}, 1);
fSeq = size(all_accs{1}, 2);
graph_accs = cell(fLetters, fSeq);
for i=1:fLetters
    for j=1:fSeq
        graph_accs{i,j} = cell(num_subj, 1);
        for z=1:num_subj
            if isempty(all_accs{z})
                continue
            end
            graph_accs{i,j}{z} = all_accs{z}{i,j};
        end
    end
end
% save("/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/results/graph_accs.mat", "graph_accs");


% load("/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/results/graph_accs.mat");
for i=10
    for j=1:fSeq
        %create figure, multiple subjects in one plot
        figure; 
        ylim([0 1]);
        hold on;
        for z=1:10
            plot(graph_accs{i,j}{z});   %plot the accuracy over the iterations
        end
        % saveas(gcf, sprintf('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/graphs/nLetters_%d.jpg', i));
        % close;
    end
end