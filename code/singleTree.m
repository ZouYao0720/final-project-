clc
clear

path = "/Users/zouyao/Documents/NEUfiles/MachineLearning/final project/train.csv";
testpath = "./final project/test.csv";
data = importdata(path);
traindata = data;
datatest = importdata(testpath);
sliddata1 = slidingData(traindata(:,1:14),4);
sliddata2 = slidingData(traindata(:,1:14),8);
sliddata3 = slidingData(traindata(:,1:14),16);

testsliddata1 = slidingData(datatest.data(:,2:15),4);
testsliddata2 = slidingData(datatest.data(:,2:15),8);
testsliddata3 = slidingData(datatest.data(:,2:15),16);

maxnode = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000];
averageErrporsingle =zeros(1,size(maxnode,2));
averageErrporFSsingle = zeros(1,size(maxnode,2));


%%%%% using 10-fold cross-validation on the training data set to evaluate
%%%%% the model
for nodeindex = 1:size(maxnode,2)     %%%% iterate the number of nodes
for i = 1:10                          %%%% 10-fold iteration
index = 1;
predictlabel = zeros(1000,20);
slice = ((i-1)*1000+1:i*1000);
    trainslice=[1:slice(1)-1,slice(end)+1:10000];
 for j = 1:size(sliddata1,3)
         treemodel = fitctree(sliddata1(trainslice,:,j),traindata(trainslice,15),'MaxNumSplits',maxnode(nodeindex));
         predictlabel(:,index) = predict(treemodel,sliddata1(slice,:,j));
        index= index+1;
 end

  for j = 1:size(sliddata2,3)
         treemodel = fitctree(sliddata2(trainslice,:,j),traindata(trainslice,15),'MaxNumSplits',maxnode(nodeindex));
         predictlabel(:,index) = predict(treemodel,sliddata2(slice,:,j));
        index = index+1;
  end
  
  treemodel = fitctree(sliddata3(trainslice,:),traindata(trainslice,15),'MaxNumSplits',maxnode(nodeindex));
   predictlabel(:,index) = predict(treemodel,sliddata3(slice,:));
  index = index+1;

  %%%%% implement the max vote principle
  for k = 1:1000
        predictlabel(k,index) = sum(predictlabel(k,:));
        if predictlabel(k,index)>=10
            predictlabel(k,index) = 1;
        else
            predictlabel(k,index) = 0;
        end
  end
   B1(:,:,i) = confusionmat(traindata(slice,15),predictlabel(:,index-1));
  B(:,:,i) = confusionmat(traindata(slice,15),predictlabel(:,index));
end






sumB1 =0;
sumB = 0;
for i = 1:10
  sumB1 = sumB1+( B1(1,2,i)+B1(2,1,i) )/1000;
  sumB = sumB+( B(1,2,i)+B(2,1,i) )/1000;
end
averageErrporsingle(nodeindex) = sumB1/10;
averageErrporFSsingle(nodeindex) = sumB/10;
end

plot(maxnode,averageErrporsingle,'r');
hold on
plot(maxnode,averageErrporFSsingle,'b');

xlabel('number of nodes in the tree','FontSize',16);
ylabel('average error of cross-validation','FontSize',16)
legend('single decision tree','FS with single decision tree','FontSize',16)
title('single decision tree vs FS-single decision tree','FontSize',16)

index =1;

%%%%%predict test
 for j = 1:size(sliddata1,3)
         treemodel = fitctree(sliddata1(:,:,j),traindata(:,15),'MaxNumSplits',500);
         testpredictlabel(:,index) = predict(treemodel,testsliddata1(:,:,j));
        index= index+1;
 end

  for j = 1:size(sliddata2,3)
         treemodel = fitctree(sliddata2(:,:,j),traindata(:,15),'MaxNumSplits',500);
        testpredictlabel(:,index) = predict(treemodel,testsliddata2(:,:,j));
        index = index+1;
  end
  
  treemodel = fitctree(sliddata3(:,:),traindata(:,15),'MaxNumSplits',500);
  testpredictlabel(:,index) = predict(treemodel,testsliddata3(:,:));
 
  
  for k = 1:4980
        testpredictlabel(k,20) = sum(testpredictlabel(k,:));
        if testpredictlabel(k,20)>=10
            testpredictlabel(k,20) = 1;
        else
            testpredictlabel(k,20) = 0;
        end
  end