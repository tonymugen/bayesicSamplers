/*
 * Copyright (c) 2018 Anthony J. Greenberg
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

/// \file Abstract base sampler class
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2018 Anthony J. Greenberg
 * \version 1.0
 *
 * Class definition and interface documentation for the abstract base MCMC sampler class. The derived classes must surface a adapt() and update() functions.
 *
 */

#pragma once

#include "../bayesicUtilities/include/random.hpp"

namespace BayesicSpace {
	/** \brief Sampler abstract base class
	 *
	 * Abstract base class for MCMC sampling methods.
	 */
	class Sampler {
	public:
		/** \brief Destructor */
		virtual ~Sampler(){};
		/** \brief Adaptation (burn-in) phase update
		 *
		 * \return Implementation-dependent exit value
		 */
		virtual int16_t adapt() = 0;
		/** \brief Sampling phase update
		 *
		 * \return Implementation-dependent exit value
		 */
		virtual int16_t update() = 0;
	protected:
		/** \brief Default constructor */
		Sampler(){};
		/** \brief Random number generator */
		RanDraw rng_;
	};
}



