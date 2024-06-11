(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e d b i)
(:init 
(harmony)
(planet e)
(planet d)
(planet b)
(planet i)
(province e)
(province d)
(province b)
(province i)
)
(:goal
(and
(craves e d)
(craves d b)
(craves b i)
)))