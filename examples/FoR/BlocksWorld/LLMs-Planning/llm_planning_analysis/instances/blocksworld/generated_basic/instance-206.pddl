

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b c)
(ontable c)
(ontable d)
(clear a)
(clear b)
(clear d)
)
(:goal
(and
(on a d)
(on b a)
(on c b))
)
)


