

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(craves c b)
(planet d)
(province a)
(province d)
)
(:goal
(and
(craves b c)
(craves c d)
(craves d a))
)
)


