/*
 * Abstract Nonlinear Least-Squares Solver Class
 *
 * nlls_solver.h
 *
 *  Created on: Nov 5, 2012
 *      Author: cforster
 */

#ifndef LM_SOLVER_IMPL_HPP_
#define LM_SOLVER_IMPL_HPP_

#include <stdexcept>

template <int D, typename T>
void vk::NLLSSolver<D, T>::optimize(ModelType& model)
{
  if(this->method_ == GaussNewton)
    optimizeGaussNewton(model);
  else if(this->method_ == LevenbergMarquardt)
    optimizeLevenbergMarquardt(model);
}

template <int D, typename T>
void vk::NLLSSolver<D, T>::optimizeGaussNewton(ModelType& model)
{
  // Compute weight scale
  if(this->use_weights_)
    computeResiduals(model, false, true);

  // Save the old model to rollback in case of unsuccessful update
  ModelType old_model(model);

  // perform iterative estimation
  for (this->iter_ = 0; this->iter_<this->n_iter_; ++this->iter_)
  {
    this->rho_ = 0;
    startIteration();

    this->H_.setZero();
    this->Jres_.setZero();

    // compute initial error
    this->n_meas_ = 0;
    double new_chi2 = computeResiduals(model, true, false);

    // add prior
    if(this->have_prior_)
      applyPrior(model);

    // solve the linear system
    if(!solve())
    {
      // matrix was singular and could not be computed
      std::cout << "Matrix is close to singular! Stop Optimizing." << std::endl;
      std::cout << "H = " << this->H_ << std::endl;
      std::cout << "Jres = " << this->Jres_ << std::endl;
      this->stop_ = true;
    }

    // check if error increased since last optimization
    if((this->iter_ > 0 && new_chi2 > this->chi2_) || this->stop_)
    {
      if(this->verbose_)
      {
        std::cout << "It. " << this->iter_
                  << "\t Failure"
                  << "\t new_chi2 = " << new_chi2
                  << "\t Error increased. Stop optimizing."
                  << std::endl;
      }
      model = old_model; // rollback
      break;
    }

    // update the model
    ModelType new_model;
    update(model, new_model);
    old_model = model;
    model = new_model;

    this->chi2_ = new_chi2;

    if(this->verbose_)
    {
      std::cout << "It. " << this->iter_
                << "\t Success"
                << "\t new_chi2 = " << new_chi2
                << "\t n_meas = " << this->n_meas_
                << "\t x_norm = " << vk::norm_max(this->x_)
                << std::endl;
    }

    finishIteration();

    // stop when converged, i.e. update step too small
    if(vk::norm_max(this->x_)<=this->eps_)
      break;
  }
}

template <int D, typename T>
void vk::NLLSSolver<D, T>::optimizeLevenbergMarquardt(ModelType& model)
{
  // Compute weight scale
  if(this->use_weights_)
    computeResiduals(model, false, true);

  // compute the initial error
  this->chi2_ = computeResiduals(model, true, false);

  if(this->verbose_)
    cout << "init chi2 = " << this->chi2_
         << "\t n_meas = " << this->n_meas_
         << endl;

  // TODO: compute initial lambda
  // Hartley and Zisserman: "A typical init value of lambda is 10^-3 times the
  // average of the diagonal elements of J'J"

  // Compute Initial Lambda
  if(this->mu_ < 0)
  {
    double H_max_diag = 0;
    double tau = 1e-4;
    for(size_t j=0; j<D; ++j)
      H_max_diag = max(H_max_diag, fabs(this->H_(j,j)));
    this->mu_ = tau*H_max_diag;
  }

  // perform iterative estimation
  for (this->iter_ = 0; this->iter_<this->n_iter_; ++this->iter_)
  {
    this->rho_ = 0;
    startIteration();

    // try to compute and update, if it fails, try with increased mu
    this->n_trials_ = 0;
    do
    {
      // init variables
      ModelType new_model;
      double new_chi2 = -1;
      this->H_.setZero();
      //H_ = mu_ * Matrix<double,D,D>::Identity(D,D);
      this->Jres_.setZero();

      // compute initial error
      this->n_meas_ = 0;
      computeResiduals(model, true, false);

      // add damping term:
      this->H_ += (this->H_.diagonal()*this->mu_).asDiagonal();

      // add prior
      if(this->have_prior_)
        applyPrior(model);

      // solve the linear system
      if(solve())
      {
        // update the model
        update(model, new_model);

        // compute error with new model and compare to old error
        this->n_meas_ = 0;
        new_chi2 = computeResiduals(new_model, false, false);
        this->rho_ = this->chi2_-new_chi2;
      }
      else
      {
        // matrix was singular and could not be computed
        cout << "Matrix is close to singular!" << endl;
        cout << "H = " << this->H_ << endl;
        cout << "Jres = " << this->Jres_ << endl;
        this->rho_ = -1;
      }

      if(this->rho_>0)
      {
        // update decrased the error -> success
        model = new_model;
        this->chi2_ = new_chi2;
        this->stop_ = vk::norm_max(this->x_)<=this->eps_;
        this->mu_ *= max(1./3., min(1.-pow(2*this->rho_-1,3), 2./3.));
        this->nu_ = 2.;
        if(this->verbose_)
        {
          cout << "It. " << this->iter_
               << "\t Trial " << this->n_trials_
               << "\t Success"
               << "\t n_meas = " << this->n_meas_
               << "\t new_chi2 = " << new_chi2
               << "\t mu = " << this->mu_
               << "\t nu = " << this->nu_
               << endl;
        }
      }
      else
      {
        // update increased the error -> fail
        this->mu_ *= this->nu_;
        this->nu_ *= 2.;
        ++this->n_trials_;
        if (this->n_trials_ >= this->n_trials_max_)
          this->stop_ = true;

        if(this->verbose_)
        {
          cout << "It. " << this->iter_
               << "\t Trial " << this->n_trials_
               << "\t Failure"
               << "\t n_meas = " << this->n_meas_
               << "\t new_chi2 = " << new_chi2
               << "\t mu = " << this->mu_
               << "\t nu = " << this->nu_
               << endl;
        }
      }

      finishTrial();

    } while(!(this->rho_>0 || this->stop_));
    if (this->stop_)
      break;

    finishIteration();
  }
}


template <int D, typename T>
void vk::NLLSSolver<D, T>::setRobustCostFunction(
    ScaleEstimatorType scale_estimator,
    WeightFunctionType weight_function)
{
  switch(scale_estimator)
  {
    case TDistScale:
      if(this->verbose_)
        printf("Using TDistribution Scale Estimator\n");
      this->scale_estimator_.reset(new robust_cost::TDistributionScaleEstimator());
      this->use_weights_=true;
      break;
    case MADScale:
      if(this->verbose_)
        printf("Using MAD Scale Estimator\n");
      this->scale_estimator_.reset(new robust_cost::MADScaleEstimator());
      this->use_weights_=true;
    break;
    case NormalScale:
      if(this->verbose_)
        printf("Using Normal Scale Estimator\n");
      this->scale_estimator_.reset(new robust_cost::NormalDistributionScaleEstimator());
      this->use_weights_=true;
      break;
    default:
      if(this->verbose_)
        printf("Using Unit Scale Estimator\n");
      this->scale_estimator_.reset(new robust_cost::UnitScaleEstimator());
      this->use_weights_=false;
  }

  switch(weight_function)
  {
    case TDistWeight:
      if(this->verbose_)
        printf("Using TDistribution Weight Function\n");
      this->weight_function_.reset(new robust_cost::TDistributionWeightFunction());
      break;
    case TukeyWeight:
      if(this->verbose_)
        printf("Using Tukey Weight Function\n");
      this->weight_function_.reset(new robust_cost::TukeyWeightFunction());
      break;
    case HuberWeight:
      if(this->verbose_)
        printf("Using Huber Weight Function\n");
      this->weight_function_.reset(new robust_cost::HuberWeightFunction());
      break;
    default:
      if(this->verbose_)
        printf("Using Unit Weight Function\n");
      this->weight_function_.reset(new robust_cost::UnitWeightFunction());
  }
}

template <int D, typename T>
void vk::NLLSSolver<D, T>::setPrior(
    const T&  prior,
    const Eigen::Matrix<double, D, D>&  Information)
{
  this->have_prior_ = true;
  this->prior_ = prior;
  this->I_prior_ = Information;
}

template <int D, typename T>
void vk::NLLSSolver<D, T>::reset()
{
  this->have_prior_ = false;
  this->chi2_ = 1e10;
  this->mu_ = this->mu_init_;
  this->nu_ = this->nu_init_;
  this->n_meas_ = 0;
  this->n_iter_ = this->n_iter_init_;
  this->iter_ = 0;
  this->stop_ = false;
}

template <int D, typename T>
inline const double& vk::NLLSSolver<D, T>::getChi2() const
{
  return this->chi2_;
}

template <int D, typename T>
inline const Eigen::Matrix<double, D, D>& vk::NLLSSolver<D, T>::getInformationMatrix() const
{
  return this->H_;
}

#endif /* LM_SOLVER_IMPL_HPP_ */
