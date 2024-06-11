

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b c)
(ontable c)
(on d b)
(clear a)
(clear d)
)
(:goal
(and
(on c b)
(on d a))
)
)


