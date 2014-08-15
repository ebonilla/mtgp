function plot_predictions(x, Y, D, Ypred, Vpred, ind_kf_train, ind_kx_train)
M = size(Y,2);

if (D==1)
    figure;
    plot(x,Y);
    hold on;
    plot(x,Ypred, 'x', 'MarkerSize', 10); % 
    for j = 1 : M 
        idx = find(ind_kf_train == j);
        xx = x(ind_kx_train(idx));
        yy = Y(ind_kx_train(idx),j);
        plot(xx,yy,'ko','markersize',10);
    end
    
    %% confidence interval plots
    for j = 1 : M
        yy = Ypred(:,j);        
        se = sqrt(Vpred(:,j));
        figure, plot_confidence_interval(x, yy, se, 1);
        hold on;
        %
        idx = find(ind_kf_train == j);
        xx  = x(ind_kx_train(idx));
        yy  = Y(ind_kx_train(idx),j);
        plot(xx,yy,'ko','markersize',10);        
    end
    
elseif (D==2)
    for j= 1: M 
        figure;
        %surf(X1,X2,reshape(Y(:,j),size(X1,1),size(X1,2))); 
        idx = find(ind_kf_train == j);
        %hold on;
        plot3(x(ind_kx_train(idx),1),x(ind_kx_train(idx),2),...
        Y(ind_kx_train(idx),j),'o','markersize',18);
        hold on;
        plot3(x(:,1),x(:,2),Ypred(:,j), 'rx', 'markersize',7);  
        grid on;
    end
else
    return;
end



