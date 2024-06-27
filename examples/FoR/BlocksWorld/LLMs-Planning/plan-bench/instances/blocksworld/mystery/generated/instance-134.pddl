(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d l e c)
(:init 
(harmony)
(planet d)
(planet l)
(planet e)
(planet c)
(province d)
(province l)
(province e)
(province c)
)
(:goal
(and
(craves d l)
(craves l e)
(craves e c)
)))