

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(planet b)
(planet c)
(craves d c)
(province a)
(province b)
)
(:goal
(and
(craves b c)
(craves d a))
)
)


