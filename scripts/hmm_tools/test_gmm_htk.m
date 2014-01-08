% Script to demonstrate the use of the MATLAB-HTK interface to train GMMs
% Generate the data
ndata = 500000;

% Fix the seeds for reproducible results
randn('state', 42);
rand('state', 42);
data = randn(ndata, 2);
prior = [0.3 0.5 0.2];
% Mixture model swaps clusters 1 and 3
datap = [0.2 0.5 0.3];
datac = [0 2; 0 0; 2 3.5];
datacov = repmat(eye(2), [1 1 3]);
data1 = data(1:prior(1)*ndata,:);
data2 = data(prior(1)*ndata+1:(prior(2)+prior(1))*ndata, :);
data3 = data((prior(1)+prior(2))*ndata +1:ndata, :);

% First cluster has axis aligned variance and centre (2, 3.5)
data1(:, 1) = data1(:, 1)*0.4 + 2.0;
data1(:, 2) = data1(:, 2)*0.8 + 3.5;
datacov(:, :, 3) = [0.4*0.4 0; 0 0.8*0.8];

% Second cluster has variance axes rotated by 30 degrees and centre (0, 0)
rotn = [cos(pi/6) -sin(pi/6); sin(pi/6) cos(pi/6)];
data2(:,1) = data2(:, 1)*0.5;
data2 = data2*rotn;
datacov(:, :, 2) = rotn' * [0.25 0; 0 1] * rotn;

% Third cluster is at (0,2)
data3 = data3 + repmat([0 2], prior(3)*ndata, 1);

% Put the dataset together again
data = [data1; data2; data3];

fh1 = figure;
plot(data(:, 1), data(:, 2), 'o')
set(gca, 'Box', 'on')

% Set up mixture model
ncentres = 3;
input_dim = 2;
mix = gmm(input_dim, ncentres, 'full');

% Initialize using matlab
% Initialise the model parameters from the data
options = foptions;
options(14) = 5;	% Just use 5 iterations of k-means in initialisation
mix = gmminit(mix, data, options);

options = zeros(1, 18);
options(1)  = 1;		% Prints out error values.
options(14) = 50;		% Number of iterations.
options(19) = true;        % Use parallel mode when applicable

tic; [mix_ip_htk, options, errlog] = gmmem_init_htk(mix, data, options); toc;

options(19) = false;
tic; [mix_i_htk, options, errlog] = gmmem_init_htk(mix, data, options); toc;
tic; [mix_p_htk, options, errlog] = gmmem_htk_parallel(mix, data, options); toc;
tic; [mix_htk, options, errlog] = gmmem_htk(mix, data, options); toc;
tic; [mix_matlab, options, errlog] = gmmem(mix, data, options); toc;

% Plot the result 
x = -4.0:0.2:5.0;
y = -4.0:0.2:5.0;
[X, Y] = meshgrid(x,y);
X = X(:);
Y = Y(:);
grid = [X Y];
Z = gmmprob(mix_htk, grid);
Z = reshape(Z, length(x), length(y));
c = mesh(x, y, Z);
hold on
title('Surface plot of probability density')
hold off
drawnow


% Try to calculate a sensible position for the second figure, below the first
fig1_pos = get(fh1, 'Position');
fig2_pos = fig1_pos;
fig2_pos(2) = fig2_pos(2) - fig1_pos(4) - 30;
fh2 = figure('Position', fig2_pos);

h3 = plot(data(:, 1), data(:, 2), 'bo');
axis equal;
hold on
title('Plot of data and covariances')
for i = 1:ncentres
  [v,d] = eig(mix_htk.covars(:,:,i));
  for j = 1:2
    % Ensure that eigenvector has unit length
    v(:,j) = v(:,j)/norm(v(:,j));
    start=mix_htk.centres(i,:)-sqrt(d(j,j))*(v(:,j)');
    endpt=mix_htk.centres(i,:)+sqrt(d(j,j))*(v(:,j)');
    linex = [start(1) endpt(1)];
    liney = [start(2) endpt(2)];
    line(linex, liney, 'Color', 'k', 'LineWidth', 3)
  end
  % Plot ellipses of one standard deviation
  theta = 0:0.02:2*pi;
  x = sqrt(d(1,1))*cos(theta);
  y = sqrt(d(2,2))*sin(theta);
  % Rotate ellipse axes
  ellipse = (v*([x; y]))';
  % Adjust centre
  ellipse = ellipse + ones(length(theta), 1)*mix_htk.centres(i,:);
  plot(ellipse(:,1), ellipse(:,2), 'r-');
end
