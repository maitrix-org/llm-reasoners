(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d a e)
(:init 
(harmony)
(planet d)
(planet a)
(planet e)
(province d)
(province a)
(province e)
)
(:goal
(and
(craves d a)
(craves a e)
)))