

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(planet c)
(craves d c)
(province a)
(province b)
(province d)
)
(:goal
(and
(craves a d)
(craves b a)
(craves c b))
)
)


