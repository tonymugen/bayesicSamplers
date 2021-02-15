/*
 * Copyright (c) 2021 Anthony J. Greenberg
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */


/// Metropolis sampler
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2021 Anthony J. Greenberg
 * \version 1.0
 *
 * A Metropolis sampler with a simple Gaussian proposal.
 *
 */

#ifndef metropolis_hpp
#define metropolis_hpp

#include <vector>

#include "../bayesicUtilities/random.hpp"
#include "model.hpp"
#include "sampler.hpp"

using std::vector;

namespace BayesicSpace {
	/** \brief Metropolis sampler
	 *
	 * Simple Metropolis sampler with a Gaussian proposal.
	 *
	 */
	class SamplerMetro final : public Sampler {
	public:
		/** \brief Default constructor */
		SamplerMetro() : Sampler(), model_{nullptr}, theta_{nullptr}, incr_{1.0} {};
		/** \brief Constructor
		 *
		 * \param[in] model pointer to a model object that has a logPost() function
		 * \param[in] theta pointer to a parameter vector
		 * \param[in] incr standard deviation of the Gaussian proposal
		 *
		 */
		SamplerMetro(const Model *model, vector<double> *theta, const double &incr) : Sampler(), model_{model}, theta_{theta}, incr_{incr} {};

		/** \brief Copy constructor (deleted) */
		SamplerMetro(const SamplerMetro &in) = delete;
		/** \brief Copy assignment operator (deleted) */
		SamplerMetro& operator=(const SamplerMetro &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to be moved
		 */
		SamplerMetro(SamplerMetro &&in);
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return Output object
		 *
		 */
		SamplerMetro& operator=(SamplerMetro &&in);

		/** \brief Destructor */
		~SamplerMetro() {model_ = nullptr; theta_ = nullptr; };

		/** \brief Adaptation step
		 *
		 * \return accept/reject indicator (1 for accept, 0 for reject)
		 *
		 */
		int16_t adapt() override;
		/** \brief Sampling step
		 *
		 * \return accept/reject indicator (1 for accept, 0 for reject)
		 *
		 */
		int16_t update() override;
	protected:
		/** \brief Pointer to a model object */
		const Model *model_;
		/** \brief Pointer to the parameter vector */
		vector<double> *theta_;
		/** \brief Gaussian proposal standard deviation (step size) */
		double incr_;
	};
}
#endif //metropolis_hpp
