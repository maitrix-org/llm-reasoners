(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d e l)
(:init 
(harmony)
(planet d)
(planet e)
(planet l)
(province d)
(province e)
(province l)
)
(:goal
(and
(craves d e)
(craves e l)
)))