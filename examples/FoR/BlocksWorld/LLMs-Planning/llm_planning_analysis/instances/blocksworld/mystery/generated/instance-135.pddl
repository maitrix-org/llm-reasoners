(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d e c g)
(:init 
(harmony)
(planet d)
(planet e)
(planet c)
(planet g)
(province d)
(province e)
(province c)
(province g)
)
(:goal
(and
(craves d e)
(craves e c)
(craves c g)
)))