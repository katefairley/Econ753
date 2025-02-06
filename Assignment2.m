% Assignment 2 Econ 753
cd('C:\Users\KatyK\University of Michigan Dropbox\Katherine Fairley\Notability\Spring 2025\753 - Methods\Github')

% Question 1

data = csvread('psychtoday.csv',1);

y = data(:,1);
X = data(:,2:end);

B0 = zeros(6,1);


%% Question 1.1 Quasi-Newton with BFGS and a numerical derivative
[B_estimated_numeric, output_numeric, time_numeric] = max_numgrad(X, y, B0)

%% Question 1.2 Quasi-Newton with BFGS and an analytical derivative
[B_estimated_analytic, output_analytic, time_analytic] = max_analyticgrad(X, y, B0)

%% Question 1.3 Nelder-Mead
obj_fun = @(B) -loglikelihood(X, y, B); % Define the objective function
options = optimset('TolX', 1e-12, 'MaxFunEvals', 1e4);

tic; % Start timer
[B_estimated_NM, fval, exitflag, output] = fminsearch(obj_fun, B0,options) % Use fminsearch to maximize ll
time_NM = toc;
output_NM = output;

%% Question 1.4
[beta_est, conv] = bhhh_poisson(X, y, B0)

%% Question 3
B = B0;
n = length(y);
k = length(B0);
tol = 1e-12;
iter_count = 0;
func_evals = 0;
tic;

% Initialize residual
r_old = Inf;
converged = false;

while ~converged
    iter_count = iter_count + 1;
    
    % Calculate residual
    f = exp(X*B);
    r = y - f;
    func_evals = func_evals + 1;
    
    % Check residual convergence
    rel_res_change = abs(norm(r) - norm(r_old))/norm(r_old);
    if rel_res_change < tol && iter_count > 1
        converged = true;
        break;
    end
    r_old = r;
    
    % Calculate the Jacobian
    J = X .* f;
    
    % Calculate the approximation of the Hessian
    H = J'*J;
    
    % Add regularization if matrix is close to singular
    if rcond(H) < 1e-12
        H = H + 1e-4 * eye(k); 
    end
    
    % Calculate step
    d = H\(J'*r);
    
    B_new = B + d;
    
    
    B = B_new;
end

toc;


%% Functions

% Maximization with numerical gradient (Q1.1)
function [B_opt, output, time] = max_numgrad(X, y, B0)
    tic;
    % Set optimization options
    options = optimoptions('fminunc', ...
        'SpecifyObjectiveGradient', false, ... % Use numerical derivatives by default
        'Algorithm', 'quasi-newton', ...
        'Display', 'iter', ...
        'StepTolerance', 1e-12);
    
    % Define objective function
    obj_fun = @(B) -loglikelihood(X, y, B);
    
    % Minimize negative log-likelihood
    [B_opt, fval, exitflag, output] = fminunc(obj_fun, B0, options)

    %iterations = output.iterations;
    %func_evals = output.funcCount;
    time = toc;
end

% Log-likelihood function
function ll = loglikelihood(X, y, B)
    ll = (-exp(X*B) + y.*X*B - log(factorial(y))).'*ones(height(y),1);
end


% Maximization with analytical gradient (Q1.2)
function [B_opt, output, time] = max_analyticgrad(X, y, B0)
   tic;
   options = optimoptions('fminunc', ...
       'SpecifyObjectiveGradient', true, ...
       'Algorithm', 'quasi-newton', ...
       'Display', 'iter', ...
       'StepTolerance', 1e-12);
   
   obj_fun = @(B) loglike_with_grad(X, y, B);
   [B_opt, fval, exitflag, output] = fminunc(obj_fun, B0, options);

   time = toc;
end

% Log-likelihood function with analytical gradient specified
function [f, g] = loglike_with_grad(X, y, B_val)
    XB = X * B_val;
    expXB = exp(XB);
    f = -(-sum(expXB) + y'*XB - sum(log(factorial(y))));
    g = -(X' * (y - expXB));
end


% Score function
function score = score_function(beta, X, y)
    score = X' * (y - exp(X*beta));
end

% Do BHHH Algorithm
function [beta, converged] = bhhh_poisson(X, y, beta_init)
   tol = 1e-6;
   beta = beta_init;
   n = length(y);
   
   iter_count = 0;
   func_evals = 0;
   tic;
   
   % Calculate initial Hessian approximation and eigenvalues
   mu_init = exp(X*beta_init);
   scores_init = zeros(n, length(beta_init));
   for i = 1:n
       scores_init(i,:) = X(i,:)' * (y(i) - mu_init(i));
   end
   B_init = scores_init' * scores_init;
   eig_init = eig(B_init);
   
   beta_new = beta + 2*tol;
   
   while norm(beta_new - beta) >= tol
       iter_count = iter_count + 1;
       beta = beta_new;
       
       mu = exp(X*beta);
       ll = (-mu + y.*X*beta - log(factorial(y))).'*ones(height(y),1);
       func_evals = func_evals + 1;
       
       scores = zeros(n, length(beta));
       for i = 1:n
           scores(i,:) = X(i,:)' * (y(i) - mu(i));
       end
       
       B = scores' * scores;
       total_score = sum(scores, 1)';
       
       if rcond(B) < 1e-12
           B = B + 1e-6 * eye(size(B));
       end
       delta = B \ total_score;
       beta_new = beta + delta;
   end
   
   % Calculate final Hessian approximation and eigenvalues
   mu = exp(X*beta_new);
   scores_fin = zeros(n, length(beta));
   for i = 1:n
       scores_fin(i,:) = X(i,:)' * (y(i) - mu(i));
   end
       
   B = scores_fin' * scores_fin;

   eig_final = eig(B);
   
   elapsed_time = toc;
   fprintf('Converged in %d iterations\n', iter_count);
   fprintf('Function evaluations: %d\n', func_evals);
   fprintf('Elapsed time: %.4f seconds\n', elapsed_time);
   fprintf('\nEigenvalues of initial Hessian approximation:\n');
   disp(eig_init);
   fprintf('\nEigenvalues of final Hessian approximation:\n');
   disp(eig_final);
   converged = true;
end

