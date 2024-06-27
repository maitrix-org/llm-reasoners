(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d h i e)
(:init 
(harmony)
(planet d)
(planet h)
(planet i)
(planet e)
(province d)
(province h)
(province i)
(province e)
)
(:goal
(and
(craves d h)
(craves h i)
(craves i e)
)))