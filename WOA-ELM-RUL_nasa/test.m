% 
% % 设置文件路径
% filePath = 'D:\embed_work\python_project\issa-lightgbm\output\PC2\cell1_pca.csv';
% 
% % 读取 CSV 文件
% dataTable = readtable(filePath);
% %%
% % 假设我们要选择的输入列名是 'InputColumn1', 'InputColumn2'
% % 输出列名是 'OutputColumn'
% inputColumns = dataTable(:, {'PC1', 'PC2'});
% outputColumn = dataTable(:, 'SOH');
% 
% 
% 
% inputArray = table2array(inputColumns);
% outputArray = table2array(outputColumn);
% 
% attributes=inputArray;
% attributes=attributes';
% strength=outputArray;
% strength=strength';
% % 查看数据
% disp(head(inputColumns));
% disp(head(outputColumn));



% dataToSave = [ELM_T_sim;WOA_ELM_T_sim0 ];

% 使用 csvwrite
% csvwrite('cell1_result.csv', dataToSave);
% clear 
% close all;

WOA_ELM_T_sim0 = cell1result(:,2)
ELM_T_sim = cell1result(:,1)
plot(WOA_ELM_T_sim0 ,'r--','LineWidth', 1.5);
% plot(ELM_T_sim ,'b-.','LineWidth', 1.5);